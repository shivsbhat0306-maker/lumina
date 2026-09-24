/**
 * The quick-search loop. Plan → choose a tool → observe → repeat → answer.
 * Written by hand, not a framework, per PRD's "the loop... Yours, not a framework's."
 *
 * How citations stay grounded (PRD's "grounded or nothing"): the model is never allowed
 * to invent a source number. Every time fetch_page succeeds, the fetched page is appended
 * to `sources` and assigned the NEXT number, and the model is shown the running numbered
 * list before every turn ("Sources so far"). It can only cite [n] for an n it has already
 * been handed. There is no code path that turns model prose into a source — only a
 * successful fetch_page call does that.
 *
 * Streaming note: this MVP calls the model non-streaming (simpler to reason about the
 * tool-use loop), then "replays" the finished answer to the client word-by-word over SSE
 * so the UI still sees it arrive progressively. Swap in Anthropic's native token streaming
 * for the final turn later if you want true token-by-token generation — the SSE contract
 * doesn't care how the tokens were produced, only that they arrive as separate `token`
 * events after `sources`.
 */
import Anthropic from '@anthropic-ai/sdk';
import type { Source, TraceEvent, AskMode, PlanEvent, SubQuestion, Locator, ToolName } from '@lumina/contract';
import { DEEP_ONLY_TOOLS } from '@lumina/contract';
import { env, secrets } from './env.js';
import { webSearch } from './tools/webSearch.js';
import { fetchPage } from './tools/fetchPage.js';
import { getCachedSearch, setCachedSearch } from './searchCache.js';
import { saveMemory, recallMemory } from './tools/memory.js';
import { searchDocuments } from './tools/searchDocuments.js';

const anthropic = new Anthropic({ apiKey: secrets.anthropic });

// Loosely typed on purpose: the exact nested type names in @anthropic-ai/sdk have moved
// between versions. If `npm run typecheck` complains here after `npm install`, tighten
// these to whatever your installed version exports — the runtime shape (role/content
// blocks with `type: "tool_use" | "text" | "tool_result"`) is stable across recent
// versions even when the type names aren't.
type AnthropicMessage = { role: 'user' | 'assistant'; content: any };

const WEB_SEARCH_TOOL = {
  name: 'web_search',
  description:
    'Search the live web. Returns a short list of {title, url, snippet}. A snippet is not enough to cite from — fetch_page the promising results before writing the answer.',
  input_schema: {
    type: 'object' as const,
    properties: { query: { type: 'string', description: 'the search query' } },
    required: ['query']
  }
};

const FETCH_PAGE_TOOL = {
  name: 'fetch_page',
  description:
    'Fetch a URL and return its main readable text. Call this before citing a page. Each page you successfully fetch is assigned the next citation number, in fetch order — cite it as [n] using exactly that number. You cannot cite a page you have not fetched.',
  input_schema: {
    type: 'object' as const,
    properties: { url: { type: 'string', description: 'the URL to fetch' } },
    required: ['url']
  }
};

const SAVE_MEMORY_TOOL = {
  name: 'save_memory',
  description:
    'Save a durable fact or preference about this user, to be recalled in FUTURE conversations — not routine detail already covered by this thread\'s own history. Use only when the user states something worth remembering long-term, e.g. "I prefer TypeScript examples" or "I live in Austin." Write it as a standalone statement, since it will be read with no other context.',
  input_schema: {
    type: 'object' as const,
    properties: { text: { type: 'string', description: 'the fact or preference to remember' } },
    required: ['text']
  }
};

const RECALL_MEMORY_TOOL = {
  name: 'recall_memory',
  description:
    "Search this user's saved long-term memories for anything relevant to the current question. Worth calling early, before answering, whenever a saved preference or past fact could change how you should respond.",
  input_schema: {
    type: 'object' as const,
    properties: { query: { type: 'string', description: 'what to search memory for' } },
    required: ['query']
  }
};

const SEARCH_DOCUMENTS_TOOL = {
  name: 'search_documents',
  description:
    "Search this user's own uploaded documents (the Space attached to this conversation) for passages relevant to the query. Unlike web_search, results are the FULL passage text, already citable — each result is assigned its own citation number immediately, with no separate fetch step needed.",
  input_schema: {
    type: 'object' as const,
    properties: { query: { type: 'string', description: 'what to search the documents for' } },
    required: ['query']
  }
};

/**
 * Step 8, deep search ONLY. Never offered alongside the other tools (see toolsForContext) —
 * it is handed to the model alone, with tool_choice forcing this exact call, as the very
 * first thing a deep run does. That is what makes "plan before any retrieval" a structural
 * guarantee instead of a hope.
 */
const PLAN_RESEARCH_TOOL = {
  name: 'plan_research',
  description:
    "Decompose the user's question into 3 to 6 sub-questions whose answers, together, would produce a genuinely thorough answer — not the same question rephrased several times. Each needs a one-line reason. Call this exactly once.",
  input_schema: {
    type: 'object' as const,
    properties: {
      subQuestions: {
        type: 'array',
        minItems: 3,
        maxItems: 6,
        items: {
          type: 'object',
          properties: {
            question: { type: 'string', description: 'one focused sub-question' },
            reason: { type: 'string', description: 'why this sub-question matters, in one line' }
          },
          required: ['question', 'reason']
        }
      }
    },
    required: ['subQuestions']
  }
};

const SYSTEM_PROMPT = `You are LUMINA, a research assistant that answers with grounded, numbered citations.

Rules:
- Never state a fact you have not retrieved in this conversation via a tool call.
- Every user message gets its own fresh retrieval, even if it repeats or closely resembles a question you already answered earlier in this thread. Do not answer from your memory of your own earlier answer instead of calling web_search / fetch_page (or search_documents) again — the point is a current, re-grounded answer each time, not a faster one from recall.
- Cite using [n], where n is a citation number you were given in "Sources so far" below. Never invent a number, and never cite a number you have not been given.
- A search snippet is not evidence. Fetch a page with fetch_page before citing it.
- search_documents results are already full text, not snippets — cite them directly, no fetch step needed.
- When search_documents is available (a Space is attached), consider whether the question is about the user's own uploaded material and prefer it over, or alongside, web_search for that part of the answer. A question that is clearly general knowledge or current events still belongs on web_search.
- Early in a new conversation, before or alongside any other tool call, call recall_memory once to check for saved preferences about how this user wants you to respond (spelling, tone, format, standing facts about them). Do this as a habit, not only when the question itself seems to call for it — a saved style preference should shape even an answer that needs no citations. Apply anything relevant it returns.
- When you have enough fetched evidence, respond with plain text and no further tool calls — that text is the final answer, in your own words, with [n] citations inline.
- If nothing relevant turned up, say so plainly and cite nothing.`;

export interface LoopEvents {
  onTrace: (t: TraceEvent) => void;
  /** Deep search only: fired exactly once, before any trace event, with the plan. Optional
   *  so runQuickLoop callers (steps 1-7) don't need to change. */
  onPlan?: (p: PlanEvent) => void;
  /** Fired exactly once, as soon as the loop knows its source list is final — i.e. right
   *  before it starts streaming the turn that writes the answer (see StreamingGrounder).
   *  This is what lets index.ts satisfy "sources before the first token" (PRD 7) even though
   *  the answer now streams live instead of arriving all at once. Optional: a path that never
   *  streams live (the `cap` path) skips it, and index.ts falls back to sending
   *  `result.sources` itself once the loop returns. */
  onSources?: (sources: Source[]) => void;
  /** Fired for each already grounding-checked piece of the answer as it is generated, in
   *  order. Optional so a caller that doesn't care about live streaming (there isn't one
   *  left, but the type stays optional for the same reason onPlan is) still compiles. */
  onToken?: (piece: string) => void;
}

export interface LoopResult {
  answerText: string;
  sources: Source[];
  terminated: 'done' | 'cap' | 'error';
  toolCalls: Array<{ name: ToolName; ok: boolean; error?: string; ms?: number }>;
  tokensIn: number;
  tokensOut: number;
  /** True only when every web_search in this run was a cache hit (AGENTS.md). No searches
   *  at all also counts as false — there is nothing to call "cached". */
  searchCached: boolean;
  /** Deep search only: how many sub-questions the plan held. Absent on a quick run — the
   *  contract's DoneEvent.subQuestions is "absent or 0 on a quick search". */
  subQuestions?: number;
  /** How many [n] citations the grounding backstop had to strip because they didn't resolve
   *  to a real source — the model cited a number it was never given. Internal only: never
   *  part of the SSE contract or the run log, just something index.ts can log a warning about.
   *  0/absent means the model's citations were already all grounded. */
  groundingIssues?: number;
  /** True when answerText was already delivered live via events.onToken (and events.onSources
   *  already fired) as it was generated — index.ts must NOT replay it again in that case.
   *  False on the `cap` path (a canned message, never streamed) and on any turn that ended
   *  the loop without ever entering the text-generating stream. */
  streamed: boolean;
}

/**
 * The grounding backstop (AGENTS.md: "### Grounding (this is the point of the assignment)" —
 * "a citation that does not resolve to something retrieved in that request is an automatic
 * fail"). SYSTEM_PROMPT already tells the model never to invent a number, but a prompt is a
 * request, not a guarantee: in practice the model sometimes borrows web_search's own local
 * "1. / 2. / 3." result numbering as if it were an already-earned citation number, without
 * ever calling fetch_page. unresolvedCitations (contract/sse.ts — "Both the agent service
 * and the bench use this") is how we catch that after the fact.
 *
 * The non-streaming version of this check (strip-after-the-fact over a whole finished
 * string) used to live here as its own function; it's gone now because every actual answer
 * path in this file is covered without it: real generation always goes through
 * StreamingGrounder below (grounded incrementally, before a piece is ever flushed), and
 * finishOnCap's canned message (further down) is built directly from `sources` itself, so
 * it can never cite a number it wasn't handed in the first place. Kept as a one-function
 * history note rather than silently deleted, since a reviewer skimming for "where does
 * grounding get enforced" would otherwise have to go looking for it.
 */

/**
 * The live-streaming grounding backstop. Real token streaming means text reaches
 * the client as it's generated — there is no "whole answer" to check before anything is
 * shown, the way finishDone/the deep synthesis step used to. This is what keeps the same
 * guarantee true anyway: a citation is only ever flushed to the caller's onPiece once its
 * closing `]` has actually arrived, so it can be checked against the (already final, by the
 * time text generation starts) source list first. Anything still open — the buffer holds
 * text from the last unmatched `[` onward — is simply held back until it resolves, so a
 * fabricated number can never slip out mid-stream.
 */
class StreamingGrounder {
  private buffer = '';
  private stripped = 0;
  private readonly validNumbers: Set<number>;

  constructor(
    sources: Source[],
    private readonly onPiece: (piece: string) => void
  ) {
    this.validNumbers = new Set(sources.map((s) => s.n));
  }

  push(delta: string): void {
    this.buffer += delta;
    this.tryFlush();
  }

  /** Call once the turn's text is done. Anything still buffered at this point is a stray
   *  `[` that never closed — not a citation, so it's safe to send through as plain text. */
  end(): void {
    if (this.buffer) {
      this.emit(this.buffer);
      this.buffer = '';
    }
  }

  get strippedCount(): number {
    return this.stripped;
  }

  private tryFlush(): void {
    const lastOpen = this.buffer.lastIndexOf('[');
    const closesAfter = lastOpen !== -1 && this.buffer.indexOf(']', lastOpen) !== -1;
    const holdFrom = lastOpen === -1 || closesAfter ? this.buffer.length : lastOpen;
    if (holdFrom === 0) return; // the whole buffer is an unresolved trailing "[..." — wait for more
    const ready = this.buffer.slice(0, holdFrom);
    this.buffer = this.buffer.slice(holdFrom);
    this.emit(ready);
  }

  private emit(text: string): void {
    const cleaned = text.replace(/\[(\d{1,3})\]/g, (match, num) =>
      this.validNumbers.has(Number(num)) ? match : (this.stripped++, '')
    );
    if (cleaned) this.onPiece(cleaned);
  }
}

/**
 * Drop `thinking`/`redacted_thinking` blocks before an assistant turn's content goes back
 * into conversation history. We don't request extended thinking, but the streaming SDK
 * (messages.stream() + finalMessage(), v0.32.1) has been observed to reconstruct a
 * malformed/empty `thinking` block on some turns; echoing that back triggers a 400 from
 * the API ("each thinking block must contain thinking") on the next turn. The loop never
 * needs raw thinking content for its own correctness, so stripping it here is safe
 * regardless of exactly why the stray block appears.
 */
function stripThinkingBlocks(content: any[]): any[] {
  return content.filter((b) => b?.type !== 'thinking' && b?.type !== 'redacted_thinking');
}

export interface HistoryTurn {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Step 4 needs to know WHO is asking (to scope save_memory/recall_memory to this user, and
 * only this user) and, optionally, which thread (so a saved memory can record where it came
 * from). Bundled into one object rather than more positional parameters, since this is the
 * kind of thing later steps (deep search) will also need to pass down to tools.
 *
 * Step 7 adds `spaceId` (which Space's documents are in scope — search_documents is only
 * offered to the model at all when this is set, since there is nothing to search without
 * it) and `mode` (from AskBody: "web" | "docs" | "auto" — decides which tools are even on
 * the table; see toolsForContext below).
 */
export interface LoopContext {
  userId: string;
  threadId?: string;
  spaceId?: string;
  mode: AskMode;
}

/**
 * This function IS the router (SPEC.md's "Should": router mode auto decides web vs.
 * documents vs. both). Rather than a separate decision step, "auto" just means "hand the
 * model every tool that could apply and let its tool choice do the deciding" — the same
 * mechanism that already decides, per question, whether to call web_search at all. Each
 * tool call's `reason` in the trace is what makes that decision inspectable afterward.
 */
function toolsForContext(ctx: LoopContext): Array<{ name: string; [k: string]: unknown }> {
  const webTools = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL];
  const docsTools = ctx.spaceId ? [SEARCH_DOCUMENTS_TOOL] : [];
  const memoryTools = [SAVE_MEMORY_TOOL, RECALL_MEMORY_TOOL];

  const tools =
    ctx.mode === 'web'
      ? [...webTools, ...memoryTools]
      : ctx.mode === 'docs'
        ? [...docsTools, ...memoryTools]
        : [...webTools, ...docsTools, ...memoryTools]; // 'auto'

  // Structural guarantee, not just a prompt: this function is the ONLY source of tools for
  // both the quick loop and every deep sub-question's gathering loop, and plan_research
  // (DEEP_ONLY_TOOLS) must never be reachable from either — only runDeepLoop's forced
  // first call ever offers it. If this ever fires, someone added it to one of the arrays
  // above by mistake.
  if (tools.some((t) => (DEEP_ONLY_TOOLS as readonly string[]).includes(t.name))) {
    throw new Error('invariant violated: a deep-only tool leaked into a shared tool list');
  }
  return tools;
}

/**
 * `history` (step 3): prior turns from this thread, oldest first, content only — no sources
 * or trace, just what was actually said. Passed in front of the new question so a follow-up
 * like "and what about its population?" has something for "its" to resolve to. Defaults to
 * empty, so step 1/2 callers (and any brand-new thread) keep working unchanged.
 */
export async function runQuickLoop(
  query: string,
  ctx: LoopContext,
  events: LoopEvents,
  history: HistoryTurn[] = []
): Promise<LoopResult> {
  const start = Date.now();
  const sources: Source[] = [];
  const toolCalls: LoopResult['toolCalls'] = [];
  const messages: AnthropicMessage[] = [
    ...history.map((turn) => ({ role: turn.role, content: turn.content })),
    { role: 'user', content: query }
  ];

  let step = 0;
  let tokensIn = 0;
  let tokensOut = 0;
  let searchCount = 0;
  let searchCacheHitCount = 0;

  while (true) {
    const elapsedSec = (Date.now() - start) / 1000;
    if (step >= env.maxToolCalls || elapsedSec >= env.maxWallClockSec) {
      return finishOnCap();
    }

    // A provider exception here is intentionally NOT caught — it propagates to the route
    // handler, which turns it into terminated:"error" and a 502. Fail loud (AGENTS.md).
    //
    // Streamed rather than .create(): a tool-deciding turn and the turn that writes the final
    // answer are the exact same call, and which one this is isn't known until content starts
    // arriving. So every turn streams, but only a turn whose FIRST content block turns out to
    // be text (never tool_use) gets ITS text forwarded live — detected below via
    // content_block_start, the same way the old code only ever showed a turn's text when
    // toolUseBlocks.length was 0. A tool-deciding turn's own text (usually none) is still
    // silently dropped either way; nothing about that behavior changed.
    const stream = anthropic.messages.stream({
      model: env.llmModel,
      max_tokens: 1024,
      system: `${SYSTEM_PROMPT}\n\n${sourcesBlock(sources)}`,
      tools: toolsForContext(ctx) as unknown as Anthropic.Tool[],
      messages: messages as Anthropic.MessageParam[]
    });

    // A plain object, not separate `let`s: TypeScript's narrowing of a `let` reassigned only
    // inside a closure (the streamEvent handler below) collapses to `never` at the
    // optional-chained uses further down, even though the runtime behavior is fine (tsx
    // doesn't type-check). Reassigning through an object property sidesteps that entirely.
    const turn: { isFinalAnswerTurn: boolean | null; grounder: StreamingGrounder | null; liveText: string } = {
      isFinalAnswerTurn: null,
      grounder: null,
      liveText: ''
    };

    stream.on('streamEvent', (event: any) => {
      if (event.type !== 'content_block_start' || turn.isFinalAnswerTurn !== null) return;
      turn.isFinalAnswerTurn = event.content_block?.type === 'text';
      if (turn.isFinalAnswerTurn) {
        // Sources are final the moment we know this turn won't call another tool — nothing
        // downstream can add to `sources` from here on. PRD 7 ("sources before the first
        // token") only holds because this fires before any onToken below ever can.
        events.onSources?.([...sources]);
        turn.grounder = new StreamingGrounder(sources, (piece) => {
          turn.liveText += piece;
          events.onToken?.(piece);
        });
      }
    });
    stream.on('text', (delta: string) => {
      if (turn.isFinalAnswerTurn && turn.grounder) turn.grounder.push(delta);
    });

    const response = await stream.finalMessage();
    turn.grounder?.end();

    tokensIn += response.usage.input_tokens;
    tokensOut += response.usage.output_tokens;

    const toolUseBlocks = response.content.filter((b: any) => b.type === 'tool_use');

    if (toolUseBlocks.length === 0) {
      // isFinalAnswerTurn is `true` here unless the response had NO content blocks at all
      // (a genuinely empty turn) — that's the only way to reach this branch without it.
      return finishDone(turn.liveText, turn.grounder?.strippedCount ?? 0, turn.isFinalAnswerTurn === true);
    }

    messages.push({ role: 'assistant', content: stripThinkingBlocks(response.content as any[]) });
    const { toolResults, endStep, searchCount: sc, searchCacheHitCount: sch } = await runToolCalls(
      toolUseBlocks,
      step,
      sources,
      ctx,
      events,
      toolCalls
    );
    step = endStep;
    searchCount += sc;
    searchCacheHitCount += sch;

    messages.push({ role: 'user', content: toolResults });
  }

  function searchCached(): boolean {
    return searchCount > 0 && searchCacheHitCount === searchCount;
  }

  function finishDone(text: string, stripped: number, streamed: boolean): LoopResult {
    return {
      answerText: text,
      sources: [...sources],
      terminated: 'done',
      toolCalls,
      tokensIn,
      tokensOut,
      searchCached: searchCached(),
      streamed,
      ...(stripped > 0 ? { groundingIssues: stripped } : {})
    };
  }

  function finishOnCap(): LoopResult {
    // Bounded and honest (PRD): a capped run says so, in its own terminated field, and gives
    // an answer built only from what was actually retrieved before the cap hit. Never
    // streamed — it's a canned message built locally from `sources`, not model output.
    const text = sources.length
      ? `I ran out of tool calls before finishing. Based on what I found so far: ${sources
          .map((s) => `[${s.n}]`)
          .join(' ')}.`
      : 'I ran out of tool calls before finding anything relevant to cite.';
    return {
      answerText: text,
      sources: [...sources],
      terminated: 'cap',
      toolCalls,
      tokensIn,
      tokensOut,
      searchCached: searchCached(),
      streamed: false
    };
  }
}

const PLAN_SYSTEM_PROMPT = `You are LUMINA's research planner. Decompose the user's question into 3 to 6 sub-questions whose answers, combined, would produce a genuinely thorough answer to the ORIGINAL question — not the same question rephrased several times, and not questions that wander off it. Each sub-question needs a one-line reason explaining why answering it contributes to the overall question. Call plan_research exactly once, with your full plan — do not call any other tool.`;

const GATHER_SYSTEM_PROMPT = `You are LUMINA, gathering evidence for ONE sub-question of a larger deep search. This turn's job is ONLY to find and cite evidence for the sub-question you were given — the final answer is written later, by a separate step, from everything gathered across every sub-question.

Rules:
- Never state a fact you have not retrieved via a tool call.
- Cite using [n] from "Sources so far" below when you reference something already found — never invent a number.
- A web search snippet is not evidence; fetch_page before citing a web page. search_documents results are already full text.
- Gather 1-3 strong pieces of evidence for THIS sub-question, then respond with a few sentences and no further tool calls — that ends gathering for this sub-question. Do not keep searching once you have enough, and do not chase the sub-question forever if nothing relevant turns up after a couple of tries — say so briefly instead.`;

/** The one call that actually writes prose for a deep answer. Every source is already
 *  globally numbered by the time this runs (dedup happened while gathering, not here), so
 *  this only has to write against a numbering it's shown — no remapping happens anywhere. */
function synthesisSystemPrompt(query: string, subQuestions: SubQuestion[], sources: Source[]): string {
  return `You are LUMINA, writing the final answer for a deep search on: "${query}"

The research plan was:
${subQuestions.map((sq) => `${sq.i}. ${sq.question}${sq.reason ? ` — ${sq.reason}` : ''}`).join('\n')}

${sourcesBlock(sources)}

Write the full answer now, in this structure:
1. A short direct-answer opening (2-4 sentences) that plainly answers the ORIGINAL question, with [n] citations.
2. One "## " heading per sub-question above, addressing it using the numbered sources — cite every claim you make.
3. A closing "## What's still unknown" section naming any sub-question the sources above don't actually answer. If everything above was genuinely covered, say so in one line instead of inventing a gap.

Rules: cite ONLY [n] numbers that appear in "Sources so far" above — never invent one, never cite a number not listed. If a sub-question has no sources listed for it at all, say so honestly in its section rather than writing content with no evidence behind it. Respond with plain text only — no further tool calls.`;
}

/**
 * Deep search (step 8). Two structural guarantees this function exists to provide, both
 * called out by AGENTS.md by name:
 *
 *   1. "Plan before any retrieval" — the FIRST model call offers ONLY plan_research, with
 *      tool_choice forcing that exact call. There is no code path from here to web_search /
 *      fetch_page / search_documents before a plan exists.
 *   2. "Merge into ONE citation numbering" — every sub-question's gathering loop shares the
 *      SAME `sources` array, and runTool's dedupe (findDupeIndex) means a source rediscovered
 *      under a second sub-question reuses its existing number instead of getting a new one.
 *      Numbers are global and deduped from the moment they're assigned — there is no separate
 *      remap step because there is nothing left to remap.
 *
 * The caps (env.maxToolCallsDeep / maxWallClockSecDeep) are a SINGLE shared budget across
 * every sub-question, not one budget each — a greedy sub-question can starve the others, and
 * that shows up honestly as terminated:"cap" with some sub-questions left with zero sources,
 * rather than being hidden.
 */
export async function runDeepLoop(
  query: string,
  ctx: LoopContext,
  events: LoopEvents,
  history: HistoryTurn[] = []
): Promise<LoopResult> {
  const start = Date.now();
  const sources: Source[] = [];
  const toolCalls: LoopResult['toolCalls'] = [];
  let step = 0;
  let tokensIn = 0;
  let tokensOut = 0;
  let searchCount = 0;
  let searchCacheHitCount = 0;
  let terminated: LoopResult['terminated'] = 'done';

  const historyMessages: AnthropicMessage[] = history.map((turn) => ({ role: turn.role, content: turn.content }));

  // Phase 1: plan. Forced tool_choice is what makes "before any retrieval" true by
  // construction rather than by asking nicely — there is no other tool in scope here.
  const planResponse = await anthropic.messages.create({
    model: env.llmModel,
    max_tokens: 1024,
    system: PLAN_SYSTEM_PROMPT,
    tools: [PLAN_RESEARCH_TOOL] as Anthropic.Tool[],
    tool_choice: { type: 'tool', name: 'plan_research' } as any,
    messages: [...historyMessages, { role: 'user', content: query }] as Anthropic.MessageParam[]
  });
  tokensIn += planResponse.usage.input_tokens;
  tokensOut += planResponse.usage.output_tokens;

  const planBlock = (planResponse.content as any[]).find((b) => b.type === 'tool_use' && b.name === 'plan_research');
  if (!planBlock) {
    // Fail loud (AGENTS.md) — a deep search with no plan is not a smaller deep search, it
    // is a bug, and pretending otherwise is exactly the kind of silent failure the caught
    // provider-exception rule exists to prevent elsewhere in this file.
    throw new Error('plan_research was not called — no plan was produced for this deep search');
  }
  const rawSubQuestions = (planBlock.input?.subQuestions ?? []) as Array<{ question: string; reason?: string }>;
  if (rawSubQuestions.length < 2) {
    throw new Error(`plan_research returned too few sub-questions (${rawSubQuestions.length}); the contract requires at least 2`);
  }
  const subQuestions: SubQuestion[] = rawSubQuestions.slice(0, 8).map((sq, idx) => ({
    i: idx + 1,
    question: sq.question,
    reason: sq.reason
  }));

  // The plan event MUST arrive before any trace event (SSE contract: "plan → trace* →
  // ..."), so onPlan fires first, and only then do we log plan_research itself as a trace
  // step — bookkeeping about the plan, not part of it.
  events.onPlan?.({ subQuestions });
  step += 1;
  toolCalls.push({ name: 'plan_research', ok: true });
  events.onTrace({
    step,
    tool: 'plan_research',
    input: { query },
    ok: true,
    ms: Date.now() - start,
    reason: `decomposed into ${subQuestions.length} sub-question${subQuestions.length === 1 ? '' : 's'}`
  });

  // Phase 2: fan-out. Sub-questions now run CONCURRENTLY. This used to be a sequential
  // `for (const sq of subQuestions)` loop where each sub-question's entire gather loop had
  // to finish before the next one even started — turning "6 sub-questions" into 6 stacked
  // rounds of LLM-round-trip-plus-maybe-tool-calls. That was the dominant cost behind the
  // deep-answer SLA miss (p95 144.6s vs a 90s target — ~6 sub-questions x ~22-24s each,
  // matching the observed numbers almost exactly). Running them concurrently instead means
  // total wall-clock is bounded by the slowest sub-question, not the sum of all of them.
  //
  // step/tokensIn/tokensOut/searchCount/searchCacheHitCount/terminated are mutable state
  // shared across every concurrent sub-question loop below, all sharing the same `sources`
  // array too (see the dedupe note above). This is safe under the same invariant this file
  // already relies on for citation numbering in runToolCalls: every read-then-mutate of
  // shared state here happens in a synchronous stretch of code with no `await` in between,
  // and JS never preempts a synchronous stretch — so two sub-questions can never interleave
  // mid-update, even with several of their network calls in flight together. In particular,
  // step numbers are RESERVED as a range (`step += toolUseBlocks.length`) synchronously,
  // right after a turn's response comes back and before any of that turn's tool calls are
  // awaited — matching the exact range runToolCalls independently assigns internally — so
  // two sub-questions can never claim overlapping step numbers, and we never reassign `step`
  // from a call's return value (which could be a stale snapshot by the time it resolves).
  //
  // Budget enforcement (`step >= env.maxToolCallsDeep` / wall-clock) still works, just more
  // organically: each sub-question's own while loop checks the shared budget before it asks
  // the model for its next turn, so a sub-question that hasn't started yet when the budget
  // is already exhausted still gets zero tool calls, same as before. The one behavior change
  // is that since all sub-questions launch together, every one of them is guaranteed at
  // least an attempt at a first turn rather than strictly-earlier sub-questions monopolizing
  // the budget before later ones are ever touched — a reasonable trade for the latency win.
  //
  // That guarantee is also exactly why the shared check alone isn't enough: since every
  // sub-question's FIRST check happens before any of them has reserved a single step, all of
  // them can see the budget as untouched and each go on to request a full turn's worth of
  // tool calls — so in practice this run's overshoot came in as one "wave" sized roughly
  // (sub-question count) x (tools per turn), not runaway, but reliably over. A per-sub-
  // question fair-share cap closes that mathematically rather than hopefully: no single
  // sub-question can ever contribute more than its even split of the total budget, so the
  // sum across all of them can never exceed maxToolCallsDeep by construction.
  const perSubQuestionCap = Math.max(1, Math.floor(env.maxToolCallsDeep / subQuestions.length));
  await Promise.all(
    subQuestions.map(async (sq) => {
      const gatherMessages: AnthropicMessage[] = [
        ...historyMessages,
        {
          role: 'user',
          content: `Sub-question ${sq.i} of ${subQuestions.length}, part of researching: "${query}"\n\n${sq.question}\n\n(Why this sub-question: ${sq.reason ?? 'part of the plan'}. Gather evidence for just THIS sub-question, then stop.)`
        }
      ];
      let myTotalSteps = 0; // local to this sub-question — no shared-state race to reason about

      while (true) {
        const elapsed = (Date.now() - start) / 1000;
        if (
          myTotalSteps >= perSubQuestionCap ||
          step >= env.maxToolCallsDeep ||
          elapsed >= env.maxWallClockSecDeep
        ) {
          terminated = 'cap';
          break;
        }

        const response = await anthropic.messages.create({
          model: env.llmModel,
          max_tokens: 1024,
          system: `${GATHER_SYSTEM_PROMPT}\n\n${sourcesBlock(sources)}`,
          tools: toolsForContext(ctx) as unknown as Anthropic.Tool[],
          messages: gatherMessages as Anthropic.MessageParam[]
        });
        tokensIn += response.usage.input_tokens;
        tokensOut += response.usage.output_tokens;

        const toolUseBlocks = (response.content as any[]).filter((b) => b.type === 'tool_use');
        if (toolUseBlocks.length === 0) break; // model decided it has enough for this sub-question

        gatherMessages.push({ role: 'assistant', content: stripThinkingBlocks(response.content as any[]) });

        const myStartStep = step;
        step += toolUseBlocks.length; // reserve this turn's step range before awaiting anything below
        myTotalSteps += toolUseBlocks.length;

        const { toolResults, searchCount: sc, searchCacheHitCount: sch } = await runToolCalls(
          toolUseBlocks,
          myStartStep,
          sources,
          ctx,
          events,
          toolCalls,
          sq.i
        );
        searchCount += sc;
        searchCacheHitCount += sch;
        gatherMessages.push({ role: 'user', content: toolResults });
      }
    })
  );

  // Phase 3: ONE synthesis call writes the whole structured answer, against the final,
  // already-deduped, globally-numbered source list — see synthesisSystemPrompt. No `tools`
  // are offered to this call at all, so unlike the quick loop's per-turn call, this one is
  // guaranteed text-only — safe to stream live from the first byte, no content_block_start
  // detection needed. `sources` is already final by this point either way (cap or done),
  // since Phase 2 is fully finished before this call is ever made.
  events.onSources?.([...sources]);
  let liveText = '';
  const grounder = new StreamingGrounder(sources, (piece) => {
    liveText += piece;
    events.onToken?.(piece);
  });
  const synthesisStream = anthropic.messages.stream({
    model: env.llmModel,
    max_tokens: 2048,
    system: synthesisSystemPrompt(query, subQuestions, sources),
    messages: [
      { role: 'user', content: 'Write the final answer now, following the structure and rules above.' }
    ] as Anthropic.MessageParam[]
  });
  synthesisStream.on('text', (delta: string) => grounder.push(delta));
  const synthesisResponse = await synthesisStream.finalMessage();
  grounder.end();

  tokensIn += synthesisResponse.usage.input_tokens;
  tokensOut += synthesisResponse.usage.output_tokens;

  return {
    answerText: liveText,
    sources: [...sources],
    terminated,
    toolCalls,
    tokensIn,
    tokensOut,
    searchCached: searchCount > 0 && searchCacheHitCount === searchCount,
    subQuestions: subQuestions.length,
    streamed: true,
    ...(grounder.strippedCount > 0 ? { groundingIssues: grounder.strippedCount } : {})
  };
}

/** Shared by the quick loop and every deep sub-question's gathering loop — the numbered
 *  list the model is shown every turn so it only ever cites an [n] it has actually been
 *  given. Takes `sources` as a parameter (rather than closing over it) so both loops can
 *  call it against their own array. */
function sourcesBlock(sources: Source[]): string {
  return sources.length === 0
    ? 'Sources so far: none yet — you must retrieve something (fetch_page or search_documents) before you can cite anything.'
    : `Sources so far:\n${sources.map((s) => `[${s.n}] ${s.title} — ${s.snippet.slice(0, 200)}`).join('\n')}`;
}

/** Is `candidate` the same underlying thing as an already-cited source? Web sources match
 *  by url; doc sources match by docId + locator (the same chunk cited twice, or across two
 *  sub-questions, should reuse its number rather than doubling up). Used by runTool so that
 *  a deep search's "merge, dedupe, renumber contiguously" (AGENTS.md) happens for free —
 *  numbers are global and deduped from the moment they're assigned, never remapped later. */
function findDupeIndex(
  sources: Source[],
  candidate: { kind: 'web' | 'doc'; url?: string; docId?: string; locator?: Locator }
): number {
  return sources.findIndex((s) => {
    if (candidate.kind === 'web') return s.kind === 'web' && s.url === candidate.url;
    return s.kind === 'doc' && s.docId === candidate.docId && JSON.stringify(s.locator ?? {}) === JSON.stringify(candidate.locator ?? {});
  });
}

/**
 * Runs one turn's worth of tool_use blocks: calls each through runTool, records its trace
 * event and toolCalls entry, and builds the tool_result messages to send back. Extracted
 * from runQuickLoop's inline loop (step 8) so runDeepLoop's per-sub-question gathering
 * loops can share the exact same per-call bookkeeping instead of duplicating it — a pure
 * restructuring, not a behavior change, for the quick path.
 *
 * `startStep`/`endStep` thread the GLOBAL step counter through explicitly (rather than a
 * shared mutable variable) since a deep run's budget is shared across every sub-question,
 * not reset per sub-question.
 */
async function runToolCalls(
  toolUseBlocks: any[],
  startStep: number,
  sources: Source[],
  ctx: LoopContext,
  events: LoopEvents,
  toolCalls: LoopResult['toolCalls'],
  subQuestion?: number
): Promise<{ toolResults: any[]; endStep: number; searchCount: number; searchCacheHitCount: number }> {
  // Every tool call in this turn now runs CONCURRENTLY instead of one at a time. Awaiting
  // each call fully before starting the next was turning "2 searches" or "4 fetches" (per
  // sla.json's own budget notes) into that many serial network round-trips, which is exactly
  // why TTFT/answer-latency blew their SLA budgets.
  //
  // This is safe for citation numbering: runTool's "assign the next citation number and push
  // into `sources`" step (fetch_page / search_documents, both `sources.length + 1` then
  // `sources.push`) is entirely synchronous with no `await` inside it. JS never preempts a
  // synchronous stretch of code between await points, so even with several runTool calls in
  // flight together, two of them can never interleave mid-way through that step — whichever
  // call's network fetch resolves first claims the next number and pushes before the next
  // one's continuation even starts. No duplicate or skipped citation numbers.
  const results = await Promise.all(
    toolUseBlocks.map(async (block) => {
      const t0 = Date.now();
      try {
        const { text, reason, cacheHit } = await runTool(block.name, block.input, sources, ctx, subQuestion);
        return { block, ok: true as const, text, reason, cacheHit, ms: Date.now() - t0 };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { block, ok: false as const, message, ms: Date.now() - t0 };
      }
    })
  );

  const toolResults: any[] = [];
  let step = startStep;
  let searchCount = 0;
  let searchCacheHitCount = 0;

  // Processed in the original block order (not completion order) so trace steps and log
  // output stay deterministic and readable, even though the calls themselves ran concurrently.
  for (const result of results) {
    step += 1;
    const { block, ms } = result;
    if (result.ok) {
      const { text, reason, cacheHit } = result;
      if (block.name === 'web_search') {
        searchCount += 1;
        if (cacheHit) searchCacheHitCount += 1;
      }
      toolCalls.push({ name: block.name, ok: true, ms });
      events.onTrace({
        step,
        tool: block.name,
        input: block.input,
        ok: true,
        ms,
        reason,
        ...(subQuestion ? { subQuestion } : {})
      });
      toolResults.push({ type: 'tool_result', tool_use_id: block.id, content: text });
    } else {
      const { message } = result;
      toolCalls.push({ name: block.name, ok: false, error: message, ms });
      events.onTrace({
        step,
        tool: block.name,
        input: block.input,
        ok: false,
        ms,
        error: message,
        ...(subQuestion ? { subQuestion } : {})
      });
      toolResults.push({
        type: 'tool_result',
        tool_use_id: block.id,
        content: `error: ${message}`,
        is_error: true
      });
    }
  }

  return { toolResults, endStep: step, searchCount, searchCacheHitCount };
}

async function runTool(
  name: string,
  input: Record<string, unknown>,
  sources: Source[],
  ctx: LoopContext,
  subQuestion?: number
): Promise<{ text: string; reason?: string; cacheHit?: boolean }> {
  if (name === 'web_search') {
    const query = String(input.query ?? '');
    const cached = await getCachedSearch(query, env.searchProvider);
    const results = cached ?? (await webSearch(query));
    if (!cached) {
      await setCachedSearch(query, env.searchProvider, results);
    }
    // Deliberately NOT "1. / 2. / 3." — a bare numbered list here looks exactly like a
    // citation number, and in testing the model cited "[1][2][3]" straight off this list
    // without ever calling fetch_page (enforceGrounding now catches that if it still
    // happens, but the point of this format is to stop inviting it in the first place).
    const text = results.length
      ? results
          .map((r) => `RESULT (uncited — call fetch_page on the url below before you may cite it)\ntitle: ${r.title}\nurl: ${r.url}\nsnippet: ${r.snippet}`)
          .join('\n\n')
      : 'no results';
    return {
      text,
      reason: `searched for "${query}"${cached ? ' (cache hit)' : ''}`,
      cacheHit: Boolean(cached)
    };
  }

  if (name === 'fetch_page') {
    const url = String(input.url ?? '');
    const page = await fetchPage(url);

    // Reuse an existing citation number for a URL already fetched (possible across
    // sub-questions in a deep run) instead of assigning a duplicate one.
    const existing = findDupeIndex(sources, { kind: 'web', url: page.url });
    let n: number;
    if (existing >= 0) {
      n = sources[existing]!.n;
    } else {
      n = sources.length + 1;
      sources.push({
        n,
        kind: 'web',
        title: page.title,
        snippet: page.text.slice(0, 500),
        url: page.url,
        ...(subQuestion ? { subQuestion } : {})
      });
    }
    return {
      text: `[${n}] ${page.title}\n${page.text.slice(0, 4000)}`,
      reason: `fetched and read ${url}, ${existing >= 0 ? 'already cited as' : 'assigned citation'} [${n}]`
    };
  }

  if (name === 'save_memory') {
    const text = String(input.text ?? '').trim();
    if (!text) {
      throw new Error('save_memory: text is required');
    }
    const id = await saveMemory(ctx.userId, text, ctx.threadId);
    return { text: `saved as memory ${id}`, reason: `saved memory: "${text}"` };
  }

  if (name === 'recall_memory') {
    const query = String(input.query ?? '');
    const hits = await recallMemory(ctx.userId, query);
    const text = hits.length
      ? hits.map((h, i) => `${i + 1}. ${h.text}`).join('\n')
      : 'no matching memories';
    return {
      text,
      reason: `recalled memory for "${query}" (${hits.length} match${hits.length === 1 ? '' : 'es'})`
    };
  }

  if (name === 'search_documents') {
    if (!ctx.spaceId) {
      // Shouldn't be reachable — toolsForContext only offers this tool when spaceId is
      // set — but fail loud rather than silently searching nothing if it ever is.
      throw new Error('search_documents called with no Space attached to this request');
    }
    const query = String(input.query ?? '');
    const hits = await searchDocuments(query, ctx.spaceId, ctx.userId);
    if (hits.length === 0) {
      return { text: 'no matching passages', reason: `searched documents for "${query}" (0 matches)` };
    }
    // Unlike fetch_page, every hit here is already full text — each gets its own citation
    // number in this ONE call, not one-at-a-time across separate tool calls. Same dedupe
    // as fetch_page: the same chunk turning up under two sub-questions reuses its number.
    const cited = hits.map((h) => {
      const existing = findDupeIndex(sources, { kind: 'doc', docId: h.docId, locator: h.locator });
      let n: number;
      if (existing >= 0) {
        n = sources[existing]!.n;
      } else {
        n = sources.length + 1;
        sources.push({
          n,
          kind: 'doc',
          title: h.title,
          snippet: h.text.slice(0, 500),
          docId: h.docId,
          locator: h.locator,
          ...(subQuestion ? { subQuestion } : {})
        });
      }
      return { n, hit: h };
    });
    const text = cited.map(({ n, hit }) => `[${n}] ${hit.title}\n${hit.text.slice(0, 4000)}`).join('\n\n');
    return {
      text,
      reason: `searched documents for "${query}" (${hits.length} match${hits.length === 1 ? '' : 'es'}, hybrid + RRF), assigned citation${hits.length === 1 ? '' : 's'} [${cited.map((c) => c.n).join(', ')}]`
    };
  }

  throw new Error(`unknown or not-yet-implemented tool: ${name}`);
}
