/**
 * Step 2: the two-tier search cache (AGENTS.md "Search" section).
 *
 * Tier 1 — an in-process LRU (a plain Map, capped at MAX_LRU_ENTRIES): near-zero latency,
 * but it disappears when the process restarts and isn't shared across multiple agent
 * processes.
 * Tier 2 — the `searchCache` Mongo collection: survives restarts, shared by every process,
 * and self-expires via a TTL index (built by the provided `scripts/create-indexes.mjs` —
 * not ours to touch) on the `expiresAt` field.
 *
 * Key = sha256(normalized query + provider), exactly as AGENTS.md specifies, so "Tavily"
 * and "  tavily  " and "TAVILY" all collapse to the same row, and so do "capital of France"
 * and "capital of france".
 */
import { createHash } from 'node:crypto';
import { COLLECTIONS } from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';
import type { WebSearchHit } from './tools/webSearch.js';

const MAX_LRU_ENTRIES = 200;
const memory = new Map<string, WebSearchHit[]>();

interface CacheRow {
  _id: string;
  provider: string;
  query: string;
  results: WebSearchHit[];
  expiresAt: Date;
  createdAt: Date;
}

function normalize(query: string): string {
  return query.trim().toLowerCase().replace(/\s+/g, ' ');
}

function cacheKey(query: string, provider: string): string {
  return createHash('sha256').update(`${normalize(query)}::${provider}`).digest('hex');
}

function rememberInMemory(key: string, hits: WebSearchHit[]): void {
  if (memory.size >= MAX_LRU_ENTRIES) {
    // A Map iterates in insertion order, so the first key it hands back is the
    // least-recently-used one — evict that to make room.
    const oldest = memory.keys().next().value;
    if (oldest !== undefined) memory.delete(oldest);
  }
  memory.set(key, hits);
}

/**
 * Returns cached hits, or null on a miss. Never throws — a cache is an optimization, not a
 * correctness requirement, so if Mongo hiccups here we just treat it as a miss and let the
 * real search run instead of failing the whole tool call.
 */
export async function getCachedSearch(query: string, provider: string): Promise<WebSearchHit[] | null> {
  const key = cacheKey(query, provider);

  const inMemory = memory.get(key);
  if (inMemory) {
    memory.delete(key); // re-insert below to mark it most-recently-used
    memory.set(key, inMemory);
    return inMemory;
  }

  try {
    const row = await (await db()).collection<CacheRow>(COLLECTIONS.searchCache).findOne({ _id: key });
    if (!row) return null;
    // Defensive: the TTL index sweeps expired rows in the background, not instantly, so a
    // request can still see a technically-expired row for a moment. Don't trust it either.
    if (new Date(row.expiresAt) <= new Date()) return null;

    rememberInMemory(key, row.results);
    return row.results;
  } catch {
    return null;
  }
}

export async function setCachedSearch(query: string, provider: string, hits: WebSearchHit[]): Promise<void> {
  const key = cacheKey(query, provider);
  rememberInMemory(key, hits);

  try {
    const now = new Date();
    const expiresAt = new Date(now.getTime() + env.searchCacheTtlSeconds * 1000);
    await (await db()).collection<CacheRow>(COLLECTIONS.searchCache).updateOne(
      { _id: key },
      { $set: { provider, query, results: hits, expiresAt, createdAt: now } },
      { upsert: true }
    );
  } catch {
    // Swallow: the in-memory tier already has it for this process; the Mongo tier is a
    // nice-to-have for surviving a restart, not something worth failing the search over.
  }
}
