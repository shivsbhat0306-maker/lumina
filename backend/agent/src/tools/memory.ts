/**
 * The save_memory / recall_memory tools (AGENTS.md "Memory" section).
 *
 * save_memory: the ONLY way a long-term memory gets written — never automatic, always a
 * deliberate choice the model makes mid-conversation. That's what "explicit" means here.
 *
 * recall_memory: semantic search, not keyword search. It turns the query into the same kind
 * of number-vector as every saved memory, then asks MongoDB Atlas Vector Search for the
 * closest ones — "closest in meaning," not "contains the same words." `filter: { userId }`
 * is what keeps one user from ever recalling another user's memories.
 */
import { COLLECTIONS, SEARCH_INDEXES, newId } from '@lumina/contract';
import { db } from '../db.js';
import { embedText } from '../embeddings.js';

export async function saveMemory(userId: string, text: string, sourceThread?: string): Promise<string> {
  const embedding = await embedText(text);
  const id = newId('mem');
  // Cast: this collection's real ids are our own prefixed strings (newId('mem')), not the
  // driver's default ObjectId — see backend/agent/src/index.ts for the fuller note on why.
  await (await db()).collection(COLLECTIONS.memories).insertOne({
    _id: id,
    userId,
    text,
    embedding,
    sourceThread,
    createdAt: new Date()
  } as any);
  return id;
}

export interface MemoryHit {
  id: string;
  text: string;
  score: number;
}

export async function recallMemory(userId: string, query: string, limit = 5): Promise<MemoryHit[]> {
  const embedding = await embedText(query);
  const results = await (await db())
    .collection(COLLECTIONS.memories)
    .aggregate([
      {
        // $vectorSearch is Atlas's operator for "find documents whose vector is nearest to
        // mine" — numCandidates is how many it scans before narrowing to `limit` results;
        // higher = more thorough but slower. filter narrows the search to this user only,
        // BEFORE ranking, so it's both a privacy boundary and a speed-up.
        $vectorSearch: {
          index: SEARCH_INDEXES.memoriesVector,
          path: 'embedding',
          queryVector: embedding,
          numCandidates: 100,
          limit,
          filter: { userId: { $eq: userId } }
        }
      },
      // Reshape each result: drop the raw _id/embedding, keep a clean {id, text, score}.
      // $meta: 'vectorSearchScore' is how you ask Atlas for the similarity score it computed.
      { $project: { _id: 0, id: '$_id', text: 1, score: { $meta: 'vectorSearchScore' } } }
    ])
    .toArray();
  return results as unknown as MemoryHit[];
}
