/**
 * The search_documents tool (step 7, hybrid retrieval).
 *
 * AGENTS.md/SPEC.md: fuse a dense ($vectorSearch, meaning) and a sparse ($search, BM25
 * keyword) search over `chunks` with Reciprocal Rank Fusion (RRF), scoped to one Space.
 *
 * Re-rank step: deliberately NOT implemented, per SPEC.md's explicit allowance for "a
 * documented reason for skipping it" instead. A cross-encoder re-rank would cost a full
 * extra model call per search_documents call; for a bootcamp-scale corpus (single-digit
 * documents, low hundreds of chunks) RRF over two independently-relevant signals already
 * gets the right chunk into the top handful almost always, and the model still sees the
 * full text of every fused result (unlike a raw search UI showing 10 blue links) — so the
 * marginal value of re-ranking here is small relative to its cost. A production version at
 * a bigger corpus size is where this tradeoff would flip.
 *
 * Unlike web_search, this returns FULL chunk text, not a snippet — the indexed chunk IS
 * the evidence (AGENTS.md: "synthesize from fetched page text or indexed chunks"), so
 * there is no separate fetch step the way there is for web results.
 */
import { COLLECTIONS, SEARCH_INDEXES, type Locator } from '@lumina/contract';
import { db } from '../db.js';
import { env } from '../env.js';
import { embedText } from '../embeddings.js';

export interface DocHit {
  chunkId: string;
  docId: string;
  title: string;
  text: string;
  locator: Locator;
}

/**
 * Runs the dense and sparse searches in parallel, fuses them with RRF, and resolves each
 * winning chunk's parent document title (chunks don't store it themselves — only docId).
 */
export async function searchDocuments(query: string, spaceId: string, userId: string): Promise<DocHit[]> {
  const embedding = await embedText(query);
  const chunks = (await db()).collection(COLLECTIONS.chunks);

  const [vectorHits, textHits] = await Promise.all([
    // Dense: nearest chunks by meaning. spaceId/userId are the only fields chunks_vector
    // declares as `type: "filter"` (scripts/indexes.json) — same constraint worker.ts's
    // read-your-write probe hit, so both are filtered here directly, not in a later $match.
    chunks
      .aggregate([
        {
          $vectorSearch: {
            index: SEARCH_INDEXES.chunksVector,
            path: 'embedding',
            queryVector: embedding,
            numCandidates: env.docsCandidatesPerMethod * 5,
            limit: env.docsCandidatesPerMethod,
            filter: { spaceId: { $eq: spaceId }, userId: { $eq: userId } }
          }
        },
        { $project: { _id: 1, docId: 1, text: 1, locator: 1 } }
      ])
      .toArray(),
    // Sparse: BM25 keyword match on chunk text, same scoping via chunks_text's token
    // fields (indexes.json), applied inside the $search compound as a `filter` clause so
    // (same reasoning as the vector side) another Space's chunks are never ranked in the
    // first place, not filtered out afterward.
    chunks
      .aggregate([
        {
          $search: {
            index: SEARCH_INDEXES.chunksText,
            compound: {
              must: [{ text: { query, path: 'text' } }],
              filter: [{ equals: { path: 'spaceId', value: spaceId } }, { equals: { path: 'userId', value: userId } }]
            }
          }
        },
        { $limit: env.docsCandidatesPerMethod },
        { $project: { _id: 1, docId: 1, text: 1, locator: 1 } }
      ])
      .toArray()
  ]);

  // Reciprocal Rank Fusion: each list contributes 1/(k + rank) per chunk (rank is
  // 0-indexed position, so the top result scores 1/(k+1)); a chunk found by BOTH searches
  // sums both contributions and floats to the top — that's the whole point of fusing two
  // differently-biased signals instead of picking one.
  const scores = new Map<string, number>();
  const byId = new Map<string, { docId: string; text: string; locator: Locator }>();
  const addRanked = (hits: Array<{ _id: unknown; docId: string; text: string; locator: Locator }>) => {
    hits.forEach((hit, rank) => {
      const id = String(hit._id);
      byId.set(id, { docId: hit.docId, text: hit.text, locator: hit.locator });
      scores.set(id, (scores.get(id) ?? 0) + 1 / (env.docsRrfK + rank + 1));
    });
  };
  addRanked(vectorHits as any);
  addRanked(textHits as any);

  const fusedIds = [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, env.docsTopK)
    .map(([id]) => id);

  if (fusedIds.length === 0) return [];

  // One more lookup: resolve each winning chunk's parent document title, since chunks only
  // carry a docId, not a human-readable title — and a citation needs the title to display.
  const docIds = [...new Set(fusedIds.map((id) => byId.get(id)!.docId))];
  const docRows = await (await db())
    .collection(COLLECTIONS.documents)
    .find({ _id: { $in: docIds } } as any)
    .toArray();
  const titleById = new Map(docRows.map((d) => [d._id as unknown as string, d.title as string]));

  return fusedIds.map((id) => {
    const chunk = byId.get(id)!;
    return {
      chunkId: id,
      docId: chunk.docId,
      title: titleById.get(chunk.docId) ?? 'document',
      text: chunk.text,
      locator: chunk.locator
    };
  });
}
