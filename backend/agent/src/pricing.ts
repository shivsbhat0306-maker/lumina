/**
 * $/token pricing for the loop's own Anthropic calls (tokensIn/tokensOut, tracked in
 * loop.ts and returned on LoopResult, consumed here to fill in `done.costUsd` — previously
 * a hardcoded 0 placeholder in index.ts).
 *
 * Scope: this prices the LLM calls that produce an answer (plan/gather/synthesis turns),
 * NOT OpenAI embedding calls (EMBEDDING_MODEL = text-embedding-3-small), which the RAG
 * ingestion worker and search_documents' query embedding also incur. At $0.02/MTok, a
 * single query's embedding costs a few thousandths of a cent — negligible next to the SLA
 * budgets this number is reported against (cost per answer <= $0.05, cost per deep answer
 * <= $0.35) — so it's left out rather than adding a second provider's pricing for a rounding
 * error's worth of accuracy.
 *
 * Rates are Anthropic's standard (non-batch, no prompt-caching) per-million-token API
 * pricing, confirmed current as of Sep 2026 against
 * platform.claude.com/docs/en/about-claude/pricing. Keyed by the exact model id LLM_MODEL
 * resolves to — if that env var is ever pointed at a model not listed here, we fail
 * loud-ish (log a warning once, report $0) rather than silently reporting a wrong number
 * for the wrong model.
 */
import pino from 'pino';
import { env } from './env.js';

const log = pino({ level: env.logLevel });

interface ModelRate {
  inputPerMTok: number;
  outputPerMTok: number;
}

const RATES: Record<string, ModelRate> = {
  'claude-sonnet-5': { inputPerMTok: 2, outputPerMTok: 10 }
};

let warnedForModel: string | null = null;

export function calcCostUsd(tokensIn: number, tokensOut: number, model: string): number {
  const rate = RATES[model];
  if (!rate) {
    // Warn once per unrecognized model, not once per request, so a misconfigured
    // LLM_MODEL doesn't spam the logs — but still surface that cost figures are unreliable
    // until this table is updated.
    if (warnedForModel !== model) {
      warnedForModel = model;
      log.warn({ model }, 'no pricing on file for this model — costUsd will report 0 until pricing.ts is updated');
    }
    return 0;
  }
  const usd = (tokensIn / 1_000_000) * rate.inputPerMTok + (tokensOut / 1_000_000) * rate.outputPerMTok;
  return Math.round(usd * 1e6) / 1e6; // 6dp: enough precision for a $0.02-ish quick answer, no float noise
}
