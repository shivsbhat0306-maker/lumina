/**
 * Step 4 needs a second AI provider, separate from Claude: OpenAI, used ONLY for embeddings
 * (turning text into a list of 1536 numbers that capture its meaning, so "semantic search"
 * — finding memories that mean something similar, not just share exact words — is possible).
 * This is the one place in the agent service that calls OpenAI; everything else is Claude.
 */
import OpenAI from 'openai';
import { env, secrets } from './env.js';

const openai = new OpenAI({ apiKey: secrets.openai });

/** Returns a 1536-number vector for `text` (EMBEDDING_DIMS in the contract). Not caught here
 *  on purpose — a bad OPENAI_API_KEY or an OpenAI outage should surface as a visible failed
 *  tool call, the same as a Tavily or Anthropic failure would, not disappear silently. */
export async function embedText(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: env.embeddingModel,
    input: text
  });
  const embedding = res.data[0];
  if (!embedding) {
    throw new Error('OpenAI returned no embedding for this input');
  }
  return embedding.embedding;
}
