/**
 * The jobs worker (step 6, part 3) — the only place in LUMINA where `pending` documents
 * actually turn into `indexed` ones. Runs as its OWN process (`npm run worker`, separate
 * terminal from `npm run dev`), never inside the request that streams someone's answer —
 * SPEC.md is explicit that parsing a big PDF on that thread would stall the SSE stream for
 * every other user, and the bench checks for exactly that.
 *
 * One job kind exists: `index_document`. Its stages, in order:
 *
 *   claim (atomic) → GridFS read → parse (pdfjs-dist, page-aware) → chunk → embed each
 *   chunk → upsert into `chunks` → READ-YOUR-WRITE PROBE → documents.status = "indexed"
 *
 * "Upserted" is not "searchable": Atlas Search indexes are eventually consistent, so the
 * probe — asking the live vector index for one of the chunks just written, and only
 * believing "indexed" once it answers — is what actually earns that status (AGENTS.md).
 *
 * Crash safety: the claim below is a single atomic findOneAndUpdate, so two worker
 * processes can never grab the same job. If this process is killed mid-job, the job's row
 * is left `status: "running"` with a `claimedAt` that stops advancing — `sweepStaleClaims`
 * below is what notices that and puts it back to `pending` so another run can retry it.
 *
 * Known simplification: a retried job re-parses/re-chunks/re-embeds from scratch rather
 * than resuming from wherever it crashed. Chunk ids are deterministic (`${docId}_${ord}`)
 * and every chunk write is an upsert, so a retry is safe (no duplicate chunks) — just not
 * free. Fine for a bootcamp-scale document; a production version would checkpoint progress
 * per chunk instead of per job.
 */
import pino from 'pino';
import { GridFSBucket, ObjectId } from 'mongodb';
import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';
import { COLLECTIONS, GRIDFS_BUCKETS, SEARCH_INDEXES } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { embedText } from './embeddings.js';

const log = pino({ level: env.logLevel });
const workerId = `wrk_${process.pid}_${Date.now().toString(36)}`;

const POLL_INTERVAL_MS = 2000;
/** A job claimed longer ago than this without finishing is assumed to belong to a dead
 *  worker, not a slow one — chosen well above how long even a large PDF should take. */
const STALE_CLAIM_MS = 5 * 60 * 1000;
/** Chunking: characters per chunk, and how much the next chunk overlaps the previous one
 *  (so a fact split across a chunk boundary is still whole in at least one chunk). */
const CHUNK_CHARS = 800;
const CHUNK_OVERLAP = 100;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------- the loop

async function main(): Promise<void> {
  log.info({ workerId }, 'jobs worker starting — polling for pending jobs');
  while (true) {
    await sweepStaleClaims();
    const job = await claimNextJob();
    if (!job) {
      await sleep(POLL_INTERVAL_MS);
      continue;
    }
    await runJob(job);
  }
}

/** Anything still `running` from a worker that died mid-job goes back to `pending`. */
async function sweepStaleClaims(): Promise<void> {
  const staleBefore = new Date(Date.now() - STALE_CLAIM_MS);
  const result = await (await db())
    .collection(COLLECTIONS.jobs)
    .updateMany(
      { status: 'running', claimedAt: { $lt: staleBefore } },
      { $set: { status: 'pending' }, $unset: { claimedAt: '', workerId: '' } }
    );
  if (result.modifiedCount > 0) {
    log.warn({ count: result.modifiedCount }, 'reclaimed stale job(s) back to pending');
  }
}

/**
 * The atomic claim from the file's own header comment, unchanged: findOneAndUpdate is a
 * single operation as far as MongoDB is concerned, so there is no window in which two
 * worker processes could both see the same job as `pending` and both grab it.
 */
async function claimNextJob(): Promise<any> {
  return (await db()).collection(COLLECTIONS.jobs).findOneAndUpdate(
    { status: 'pending' },
    { $set: { status: 'running', claimedAt: new Date(), workerId }, $inc: { attempts: 1 } },
    { sort: { createdAt: 1 }, returnDocument: 'after' }
  );
}

async function runJob(job: any): Promise<void> {
  const jobId = job._id as string;
  log.info({ jobId, kind: job.kind }, 'claimed job');
  try {
    if (job.kind !== 'index_document') {
      throw new Error(`unknown job kind: ${job.kind}`);
    }
    await indexDocument(job.payload);
    await (await db()).collection(COLLECTIONS.jobs).updateOne({ _id: jobId } as any, { $set: { status: 'done' } });
    log.info({ jobId }, 'job done');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.error({ jobId, err }, 'job failed');
    await (await db())
      .collection(COLLECTIONS.jobs)
      .updateOne({ _id: jobId } as any, { $set: { status: 'failed', error: message } });
    // Fail loud on the document too, not just the job — GET /spaces/:id/documents is how
    // the UI (and you, curling it) finds out something went wrong, not the worker's logs.
    if (job.payload?.docId) {
      await (await db())
        .collection(COLLECTIONS.documents)
        .updateOne({ _id: job.payload.docId } as any, { $set: { status: 'failed', error: message } });
    }
  }
}

// ---------------------------------------------------------------- index_document

type IndexPayload = { docId: string; spaceId: string; userId: string; fileId: string; mimeType: string };

async function indexDocument(payload: IndexPayload): Promise<void> {
  const { docId, spaceId, userId, fileId, mimeType } = payload;
  const documents = (await db()).collection(COLLECTIONS.documents);

  // Stage 1: pull the raw bytes back out of GridFS — the same bucket the upload route wrote
  // them into, found by the fileId the upload route saved on the document row.
  const bucket = new GridFSBucket(await db(), { bucketName: GRIDFS_BUCKETS.uploads });
  const buffer = await streamToBuffer(bucket.openDownloadStream(new ObjectId(fileId)));

  // Stage 2: parse into "sections" — one per page for a PDF, one per heading/block for
  // text — each carrying the locator a citation will eventually point at.
  // Cast: this collection's real ids are our own prefixed strings (newId('doc')), not the
  // driver's default ObjectId — see backend/agent/src/index.ts for the fuller note.
  await documents.updateOne({ _id: docId } as any, { $set: { status: 'parsing', pct: 10 } });
  const sections =
    mimeType === 'application/pdf' ? await parsePdf(buffer) : parsePlainOrMarkdown(buffer.toString('utf8'), mimeType);
  if (sections.length === 0) {
    throw new Error('no extractable text in this document');
  }

  // Stage 3: split each section into embedding-sized chunks, numbered in reading order.
  const chunks: Array<{ text: string; locator: Record<string, unknown>; ord: number }> = [];
  let ord = 0;
  for (const section of sections) {
    for (const piece of splitIntoChunks(section.text)) {
      chunks.push({ text: piece, locator: section.locator, ord: ord++ });
    }
  }

  await documents.updateOne(
    { _id: docId } as any,
    { $set: { pct: 30, ...(mimeType === 'application/pdf' ? { pages: sections.length } : {}) } }
  );
  await documents.updateOne({ _id: docId } as any, { $set: { status: 'embedding', pct: 50 } });

  // Stage 4: embed + upsert each chunk, one at a time. Deterministic chunk ids
  // (`${docId}_${ord}`) plus upsert is what makes a re-run after a crash safe (see the
  // file header) instead of piling up duplicate chunks.
  const chunksCollection = (await db()).collection(COLLECTIONS.chunks);
  let firstEmbedding: number[] | null = null;
  for (const chunk of chunks) {
    const embedding = await embedText(chunk.text);
    if (chunk.ord === 0) firstEmbedding = embedding;
    await chunksCollection.updateOne(
      { _id: `${docId}_${chunk.ord}` } as any,
      {
        $set: {
          docId,
          spaceId,
          userId,
          text: chunk.text,
          locator: chunk.locator,
          ord: chunk.ord,
          embedding,
          createdAt: new Date()
        }
      },
      { upsert: true }
    );
    // pct climbs 50 -> 90 across all chunks, so a big document still shows live progress
    // instead of sitting at one number for however long embedding takes.
    const pct = 50 + Math.round((40 * (chunk.ord + 1)) / chunks.length);
    await documents.updateOne({ _id: docId } as any, { $set: { pct } });
  }
  await documents.updateOne({ _id: docId } as any, { $set: { chunks: chunks.length, pct: 95 } });

  // Stage 5: the read-your-write probe. Reuses chunk 0's embedding rather than paying for
  // another OpenAI call — querying with the exact vector we just wrote should be the
  // easiest possible match for the index to find once it has caught up.
  if (!firstEmbedding) {
    throw new Error('no chunks were produced to probe with');
  }
  const found = await probeUntilFound(docId, spaceId, firstEmbedding);
  if (!found) {
    throw new Error('read-your-write probe timed out: chunks were written but never became searchable');
  }

  await documents.updateOne({ _id: docId } as any, { $set: { status: 'indexed', pct: 100 } });
}

/**
 * Retries a live $vectorSearch for this docId's own chunk with backoff, since Atlas Search
 * indexing lags the write by anywhere from under a second to several seconds.
 *
 * Filters on `spaceId`, not `docId` — the chunks_vector index (scripts/indexes.json) only
 * declares spaceId and userId as `type: "filter"` fields; Atlas rejects filtering inside
 * $vectorSearch on any path the index wasn't told to index that way. So this pulls back a
 * few of the nearest candidates in the Space and checks in code whether our own docId is
 * among them — safe because we're probing with the EXACT embedding of a chunk we just
 * wrote, so it should score at or near the very top regardless.
 */
async function probeUntilFound(docId: string, spaceId: string, queryVector: number[]): Promise<boolean> {
  // Atlas Search catch-up on a shared/free-tier cluster can occasionally run past 15s under
  // load, so this gives it real headroom instead of giving up right when it's about to land.
  const delaysMs = [500, 1000, 2000, 4000, 8000, 8000, 8000, 8000]; // ~31.5s between attempts, ~8 tries
  const chunksCollection = (await db()).collection(COLLECTIONS.chunks);
  for (let i = 0; i < delaysMs.length; i++) {
    const hits = await chunksCollection
      .aggregate([
        {
          $vectorSearch: {
            index: SEARCH_INDEXES.chunksVector,
            path: 'embedding',
            queryVector,
            numCandidates: 50,
            limit: 5,
            filter: { spaceId: { $eq: spaceId } }
          }
        },
        { $project: { _id: 0, docId: 1 } }
      ])
      .toArray();
    if (hits.some((h) => h.docId === docId)) return true;
    if (i < delaysMs.length - 1) await sleep(delaysMs[i]!);
  }
  return false;
}

// ---------------------------------------------------------------- parsing

async function parsePdf(buffer: Buffer): Promise<Array<{ text: string; locator: { page: number } }>> {
  const doc = await getDocument({ data: new Uint8Array(buffer) }).promise;
  const sections: Array<{ text: string; locator: { page: number } }> = [];
  for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
    const page = await doc.getPage(pageNum);
    const content = await page.getTextContent();
    const text = content.items
      .map((item: any) => ('str' in item ? item.str : ''))
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();
    if (text) sections.push({ text, locator: { page: pageNum } });
  }
  return sections;
}

function parsePlainOrMarkdown(
  text: string,
  mimeType: string
): Array<{ text: string; locator: { heading: string } | { line: number } }> {
  if (mimeType === 'text/markdown') {
    // One section per markdown heading (# through ######) — locator is the heading text
    // itself, so a citation can read "under 'Setup'" instead of a bare line number.
    const lines = text.split(/\r?\n/);
    const sections: Array<{ text: string; locator: { heading: string } }> = [];
    let heading = '(document start)';
    let buffer: string[] = [];
    const flush = () => {
      const body = buffer.join('\n').trim();
      if (body) sections.push({ text: body, locator: { heading } });
      buffer = [];
    };
    for (const line of lines) {
      const match = line.match(/^#{1,6}\s+(.*)/);
      if (match) {
        flush();
        heading = match[1]!.trim();
      } else {
        buffer.push(line);
      }
    }
    flush();
    return sections;
  }

  // Plain text: one section per 40-line block, locator is the 1-indexed starting line.
  const lines = text.split(/\r?\n/);
  const sections: Array<{ text: string; locator: { line: number } }> = [];
  const blockSize = 40;
  for (let i = 0; i < lines.length; i += blockSize) {
    const block = lines.slice(i, i + blockSize).join('\n').trim();
    if (block) sections.push({ text: block, locator: { line: i + 1 } });
  }
  return sections;
}

/** Splits one section's text into <=CHUNK_CHARS pieces, each overlapping the previous one
 *  by CHUNK_OVERLAP characters. Character-based, not word- or sentence-aware — simple, and
 *  good enough at this chunk size that a cut mid-sentence rarely loses the actual fact. */
function splitIntoChunks(text: string): string[] {
  if (text.length <= CHUNK_CHARS) return [text];
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + CHUNK_CHARS, text.length);
    chunks.push(text.slice(start, end));
    if (end === text.length) break;
    start = end - CHUNK_OVERLAP;
  }
  return chunks;
}

// ---------------------------------------------------------------- small helper

function streamToBuffer(stream: NodeJS.ReadableStream): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const parts: Buffer[] = [];
    stream.on('data', (chunk) => parts.push(chunk as Buffer));
    stream.on('end', () => resolve(Buffer.concat(parts)));
    stream.on('error', reject);
  });
}

void main();
