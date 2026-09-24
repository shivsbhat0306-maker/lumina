import { MongoClient, type Db } from 'mongodb';
import { env } from './env.js';

let client: MongoClient | null = null;

/** One client per process. The driver pools connections; do not open one per request. */
export async function db(): Promise<Db> {
  if (!env.mongoUri) throw new Error('MONGODB_URI is not set — copy .env.example to .env');
  if (!client) {
    const candidate = new MongoClient(env.mongoUri, { serverSelectionTimeoutMS: 5000 });
    try {
      await candidate.connect();
    } catch (err) {
      // Don't leave `client` pointing at a MongoClient that never actually connected — a
      // transient failure here (network blip, an IP not yet allowlisted) would otherwise
      // wedge every future call into reusing that same broken client forever, since
      // `if (!client)` above would never be true again for the rest of the process's life.
      await candidate.close().catch(() => {});
      throw err;
    }
    client = candidate;
  }
  return client.db(env.mongoDb);
}

export async function pingDb(): Promise<'ok' | 'down'> {
  try {
    await (await db()).command({ ping: 1 });
    return 'ok';
  } catch {
    return 'down';
  }
}
