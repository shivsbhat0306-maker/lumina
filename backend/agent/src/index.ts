/**
 * LUMINA agent service — the AI backend. PROVIDED SKELETON: YOU BUILD THIS OUT.
 * This is where the real work is. Provider keys live only in this process.
 *
 * What is already here: the server, /health (Mongo ping + which model, provider and
 * vector backend are live), and a 501 for every other route.
 *
 * What you build (README Part 1, in this order — each step is testable with curl -N):
 *   1. the QUICK loop: plan → choose tool → observe → repeat → answer, with web_search
 *      and fetch_page, streaming trace → sources → token → done. sources BEFORE the
 *      first token. Disable compression on this route and flush after every event.
 *   2. the search cache: in-process LRU over the searchCache collection (TTL index),
 *      key = sha256(normalized query + provider). searchCached only when every hit.
 *   3. threads + messages, so a follow-up sees the thread.
 *   4. memory: save_memory / recall_memory over the memories vector index; GET /memory,
 *      DELETE /memory/:id.
 *   5. the run log: one runs/<requestId>.json per answer, in the RunLog shape from the
 *      contract. Ten lines. The gates read it, so it is not optional.
 *   6. spaces + the jobs worker: upload → GridFS → parse → chunk → embed → upsert →
 *      read-your-write probe → indexed.
 *   7. hybrid retrieval: $vectorSearch + $search fused with RRF, page locators.
 *   8. DEEP search (depth: "deep"): plan_research decomposes the question into 3–6
 *      sub-questions, you stream a `plan` event BEFORE retrieving anything, research each
 *      sub-question, then merge the results into ONE citation numbering and synthesise.
 *      Every trace step and every source carries the subQuestion it served. Deep runs
 *      under the wider caps (maxToolCallsDeep, maxWallClockSecDeep) and behind
 *      DEEP_DAILY_CAP → 429 {error, resetsAt}.
 *
 * Three rules to hold on to while you write it:
 *   - Fail loud. A provider exception ends the run with terminated:"error" and a 502.
 *     Never a try/catch that returns a plausible answer. (Live Translate served English
 *     for weeks because of exactly that catch.)
 *   - Grounded or nothing. A citation that does not resolve to something retrieved in
 *     THIS request is an automatic fail.
 *   - Depth is opted into, never drifted into. A quick search may not call plan_research,
 *     however much the model would like to. Deep costs several times more, and a product
 *     that escalates itself is a product with an unbounded bill.
 *
 * STATUS: steps 1–7 done. Threads (POST/GET /threads, GET /threads/:id), memory (GET /memory,
 * DELETE /memory/:id), and POST /threads/:threadId/ask (quick only) are implemented below,
 * ahead of the generic 501 loop so they take priority. web_search goes through the two-tier
 * cache in searchCache.ts; /ask loads prior messages as context and saves both sides of the
 * conversation; the loop can now also call save_memory / recall_memory (tools/memory.ts),
 * scoped to the requesting user via LoopContext. Every quick answer also writes
 * runs/<requestId>.json (RunLog shape, db.ts) after the SSE stream ends. Spaces + RAG
 * ingestion are live end to end: POST/GET /spaces, POST/GET /spaces/:spaceId/documents
 * (upload → GridFS → pending `documents`/`jobs` rows, <300ms, no parsing inline), and the
 * jobs worker (worker.ts, run separately via `npm run worker`) that turns `pending` into
 * `indexed`. Verified end to end on both a plain-text file and a 254-page PDF (773 chunks).
 * /ask now validates spaceId ownership (404 if unknown/foreign, 400 if mode "docs" with no
 * spaceId) and passes spaceId/mode into LoopContext; the loop's search_documents tool
 * (tools/searchDocuments.ts) fuses $vectorSearch + $search chunks via RRF, citing full
 * chunk text with page/heading/line locators — toolsForContext (loop.ts) is the "router":
 * mode picks which tools the model even sees, and IS the auto-routing behaviour, not a
 * separate decision step. Deep search is now live: depth:"deep" runs runDeepLoop (loop.ts)
 * instead of runQuickLoop — same route, same SSE machinery, but a `plan` event streams
 * first (forced plan_research call), every trace/source is tagged with its subQuestion, and
 * citations are globally deduped as they're gathered (no separate merge/renumber pass).
 * Guarded by DEEP_DAILY_CAP: exceeding it 429s with {error, resetsAt} before any LLM call.
 * Both quick and deep answers now write a row to the `requests` collection (and a pino
 * 'answer' log line) — that is what the daily cap counts against and what GET /stats reads.
 */
import express from 'express';
import pino from 'pino';
import multer from 'multer';
import { GridFSBucket } from 'mongodb';
import { Readable } from 'node:stream';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  HealthResponse,
  ROUTES,
  USER_HEADER,
  REQUEST_HEADER,
  AskBody,
  CreateSpaceBody,
  GRIDFS_BUCKETS,
  ACCEPTED_UPLOAD_TYPES,
  MAX_UPLOAD_BYTES,
  type CreateThreadResponse,
  type ListThreadsResponse,
  type GetThreadResponse,
  type ListMemoryResponse,
  type CreateSpaceResponse,
  type ListSpacesResponse,
  type UploadDocumentResponse,
  type ListDocumentsResponse,
  type DocStatus,
  type DoneEvent,
  type RunLog,
  type StatsResponse,
  newId
} from '@lumina/contract';
import { env } from './env.js';
import { calcCostUsd } from './pricing.js';
import { pingDb, db } from './db.js';
import { runQuickLoop, runDeepLoop, type HistoryTurn } from './loop.js';

// Step 6: files are held in memory only long enough to stream them into GridFS — fine up
// to MAX_UPLOAD_BYTES (25MB). fileSize here is what turns "file too big" into a 413
// (handled in the error middleware at the bottom of this file) instead of a crash.
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_UPLOAD_BYTES } });

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

mkdirSync(env.runsDir, { recursive: true });

// ---------------------------------------------------------------- /health (implemented)

app.get('/health', async (_req, res) => {
  const dbStatus = await pingDb();
  const body: HealthResponse = {
    status: dbStatus === 'ok' ? 'ok' : 'degraded',
    model: env.llmModel,
    searchProvider: env.searchProvider,
    vectorStore: env.vectorBackend,
    db: dbStatus,
    ai: { status: 'ok' }
  };
  res.status(dbStatus === 'ok' ? 200 : 503).json(body);
});

// ---------------------------------------------------------------- GET /stats (step 8)
// Reads back the `requests` rows written at the end of every /ask above. Per-user, like
// everything else here (AGENTS.md: deepToday/deepDailyCap are "per X-User-Id").

/** The 95th-percentile of a list of millisecond durations. Empty input reads as 0, not NaN —
 *  a user with no answers yet has a real, reportable p95: zero. */
function p95(msValues: number[]): number {
  if (msValues.length === 0) return 0;
  const sorted = [...msValues].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1);
  return sorted[idx] ?? 0;
}

app.get('/stats', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }

  const rows = await (await db()).collection('requests').find({ userId }).toArray();
  const answers = rows.filter((r) => r.status === 200);
  const todayStart = startOfTodayUTC();
  const todayRows = rows.filter((r) => new Date(r.createdAt) >= todayStart);

  const body: StatsResponse = {
    // "requests" = every /ask attempt (success or failure); "answers" = the ones that
    // actually finished with a 200 — the same distinction the pino 'answer' log line draws.
    requests: rows.length,
    answers: answers.length,
    searchCacheHitRatePct:
      answers.length === 0
        ? 0
        : Math.round((answers.filter((r) => r.searchCached === true).length / answers.length) * 1000) / 10,
    ttftP95Ms: p95(answers.map((r) => r.ttftMs as number).filter((n) => typeof n === 'number')),
    costUsdToday: todayRows.reduce((sum, r) => sum + ((r.costUsd as number | undefined) ?? 0), 0),
    deepToday: await countDeepToday(userId),
    deepDailyCap: env.deepDailyCap
  };
  res.json(body);
});

// ---------------------------------------------------------------- small helper: SSE frames

function sseWrite(res: express.Response, event: string, data: unknown) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

// MessageDoc._id (packages/contract/src/db.ts) is a plain z.string(), unlike ThreadId/SpaceId/
// DocId — but newId()'s prefix union (ids.ts) was never extended to include 'msg', so it has
// to be generated inline here with the exact same scheme, the same way jobId already is below.
function newMessageId(): string {
  return `msg_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

// ---------------------------------------------------------------- POST /threads (minimal)
// Full "threads + messages, follow-ups see history" is step 3. For now this just creates
// enough of a thread for /ask to have somewhere to point at, and writes a real ThreadDoc
// so step 3 has something to build on rather than starting over.

app.post('/threads', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }
  const threadId = newId('thr');
  // Cast: this collection's real ids are our own prefixed strings (newId('thr'), 'spc_…',
  // 'doc_…', 'mem_…', 'msg_…', 'job_…'), not the MongoDB driver's default ObjectId — the
  // whole point of ids.ts's prefixed-string scheme (PRD 8) is that an id is readable in a
  // log line or a URL. `.collection()` has no generic type argument anywhere in this file,
  // so the driver falls back to assuming `_id: ObjectId`; rather than fight that with a
  // parallel set of typed collection interfaces (real work, no behavior change either way),
  // every _id-touching call here and in worker.ts/tools/*.ts is cast at the boundary — the
  // runtime shape has always matched packages/contract/src/db.ts's Zod schemas either way.
  await (await db()).collection('threads').insertOne({
    _id: threadId,
    userId,
    title: (req.body?.title as string | undefined) ?? 'New thread',
    createdAt: new Date()
  } as any);
  const body: CreateThreadResponse = { threadId };
  res.status(201).json(body);
});

// ---------------------------------------------------------------- GET /threads (list, step 3)

app.get('/threads', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }
  const rows = await (await db())
    .collection('threads')
    .find({ userId })
    .sort({ createdAt: -1 })
    .toArray();
  const body: ListThreadsResponse = {
    threads: rows.map((t) => ({
      threadId: t._id as unknown as string,
      title: t.title as string,
      createdAt: new Date(t.createdAt).toISOString()
    }))
  };
  res.json(body);
});

// ---------------------------------------------------------------- GET /threads/:id (history, step 3)

app.get('/threads/:threadId', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }
  const { threadId } = req.params;

  // 404 unknown ids (AGENTS.md contract rule) — and "unknown" includes "exists, but
  // belongs to a different user": we scope the lookup by userId too, not just _id, so one
  // user can never read another user's thread by guessing an id.
  const thread = await (await db()).collection('threads').findOne({ _id: threadId, userId } as any);
  if (!thread) {
    return res.status(404).json({ error: `no thread ${threadId}`, status: 404 });
  }

  const messageRows = await (await db())
    .collection('messages')
    .find({ threadId, userId })
    .sort({ createdAt: 1 })
    .toArray();

  const body: GetThreadResponse = {
    threadId: thread._id as unknown as string,
    title: thread.title as string,
    messages: messageRows.map((m) => ({
      role: m.role,
      content: m.content,
      sources: m.sources,
      answerId: m.answerId,
      done: m.done,
      createdAt: new Date(m.createdAt).toISOString()
    }))
  };
  res.json(body);
});

// ---------------------------------------------------------------- deep search's daily cap (step 8)
// AGENTS.md: DEEP_DAILY_CAP is enforced HERE, in the agent service, not the gateway — the
// gateway has no idea what a "deep" run costs, only this service does. It reuses the
// `requests` collection (RequestDoc, db.ts) that was scaffolded from step 1 but never
// written to until now — the same rows also feed GET /stats just below.

/** Midnight UTC, today. The cap resets on a UTC day boundary, not the caller's local time. */
function startOfTodayUTC(): Date {
  const now = new Date();
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
}

function startOfTomorrowUTC(): Date {
  return new Date(startOfTodayUTC().getTime() + 24 * 60 * 60 * 1000);
}

async function countDeepToday(userId: string): Promise<number> {
  return (await db())
    .collection('requests')
    .countDocuments({ userId, depth: 'deep', createdAt: { $gte: startOfTodayUTC() } });
}

// ---------------------------------------------------------------- POST /threads/:id/ask (quick + deep)

app.post('/threads/:threadId/ask', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }

  const { threadId } = req.params;
  const thread = await (await db()).collection('threads').findOne({ _id: threadId, userId } as any);
  if (!thread) {
    return res.status(404).json({ error: `no thread ${threadId}`, status: 404 });
  }

  // Step 5: reused if the caller (eventually the gateway) sent one, else generated here —
  // this is what names runs/<requestId>.json below, and what a log line would grep on.
  const requestId = req.header(REQUEST_HEADER) ?? newId('req');

  const parsed = AskBody.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.message, status: 400 });
  }
  const { query, depth, mode, spaceId } = parsed.data;

  // Step 8: the daily deep-search spend gate. Checked before any other work — before even
  // the history/message-history Mongo write below — so a capped user gets a fast, cheap 429
  // instead of paying for a Mongo round trip first. resetsAt is the next UTC midnight, so a
  // client can show "resets in Xh" without the server exposing its timezone logic.
  if (depth === 'deep') {
    const deepToday = await countDeepToday(userId);
    if (deepToday >= env.deepDailyCap) {
      return res.status(429).json({
        error: `daily deep search cap (${env.deepDailyCap}) reached`,
        status: 429,
        resetsAt: startOfTomorrowUTC().toISOString()
      });
    }
  }

  // Step 7: mode "docs" is meaningless without a Space to search, so reject it before
  // doing any other work. mode "auto"/"web" tolerate a missing spaceId fine (loop.ts's
  // toolsForContext just won't offer search_documents).
  if (mode === 'docs' && !spaceId) {
    return res.status(400).json({ error: 'mode "docs" requires spaceId', status: 400 });
  }
  // Same ownership check as threads/memory: an unknown OR someone-else's spaceId 404s,
  // rather than silently searching nothing (which would look like "no relevant docs"
  // instead of "you don't have access to that Space").
  if (spaceId) {
    const space = await (await db()).collection('spaces').findOne({ _id: spaceId, userId } as any);
    if (!space) {
      return res.status(404).json({ error: `no space ${spaceId}`, status: 404 });
    }
  }

  // Step 3, part 1: load this thread's prior turns (so a follow-up question has context),
  // and record the user's new question — both before we commit to the SSE stream, so a
  // Mongo problem here can still be a plain 502 JSON response instead of a broken stream.
  let history: HistoryTurn[];
  try {
    const historyRows = await (await db())
      .collection('messages')
      .find({ threadId, userId })
      .sort({ createdAt: 1 })
      .toArray();
    history = historyRows.map((m) => ({
      role: m.role as 'user' | 'assistant',
      content: m.content as string
    }));

    await (await db()).collection('messages').insertOne({
      _id: newMessageId(),
      threadId,
      userId,
      role: 'user',
      content: query,
      sources: [],
      createdAt: new Date()
    } as any);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log.error({ err, threadId }, 'failed to load or save thread history');
    return res.status(502).json({ error: message, status: 502 });
  }

  // SSE setup. Disable compression/buffering so each write reaches the browser immediately
  // instead of sitting in a buffer until the response ends.
  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const start = Date.now();
  let ttftMs = 0;
  let firstTokenSent = false;
  let sourcesSent = false;

  try {
    // Step 8: same route, same SSE machinery, same LoopResult shape either way — only which
    // loop function runs, and whether a `plan` event ever fires, differ by depth. onPlan is
    // simply never called by runQuickLoop, so quick behaviour is unchanged by wiring it in.
    //
    // Real streaming (not the old "wait for the whole answer, then replay it word-by-word"):
    // onSources/onToken fire live, from inside the loop, as the model actually generates the
    // answer — loop.ts's StreamingGrounder is what keeps every citation still checked against
    // the real source list before a piece is ever handed to onToken, so going live cost
    // nothing on the grounding guarantee.
    const runLoop = depth === 'deep' ? runDeepLoop : runQuickLoop;
    const result = await runLoop(
      query,
      { userId, threadId, spaceId, mode },
      {
        onTrace: (t) => sseWrite(res, 'trace', t),
        onPlan: (p) => sseWrite(res, 'plan', p),
        onSources: (s) => {
          // sources MUST arrive before the first token — a contract rule (PRD 7), not a style
          // preference. Guarded so a later, redundant call (there shouldn't be one) can't
          // double-send.
          if (sourcesSent) return;
          sourcesSent = true;
          sseWrite(res, 'sources', s);
        },
        onToken: (piece) => {
          if (!firstTokenSent) {
            ttftMs = Date.now() - start;
            firstTokenSent = true;
          }
          sseWrite(res, 'token', { text: piece });
        }
      },
      history
    );

    // Fallback for a path that never streamed live — currently only `terminated: "cap"`,
    // whose answer is a canned message built from `sources`, not model output, so there was
    // never a stream to forward from. Sent instantly, no artificial pacing: it's a short
    // message, and there's no SLA reason to slow it down on purpose.
    if (!sourcesSent) {
      sourcesSent = true;
      sseWrite(res, 'sources', result.sources);
    }
    if (!result.streamed) {
      for (const piece of result.answerText.split(/(\s+)/).filter(Boolean)) {
        if (!firstTokenSent) {
          ttftMs = Date.now() - start;
          firstTokenSent = true;
        }
        sseWrite(res, 'token', { text: piece });
      }
    }

    const done: DoneEvent = {
      answerId: newId('ans'),
      latencyMs: Date.now() - start,
      ttftMs: firstTokenSent ? ttftMs : Date.now() - start,
      model: env.llmModel,
      tokens: { in: result.tokensIn, out: result.tokensOut },
      costUsd: calcCostUsd(result.tokensIn, result.tokensOut, env.llmModel),
      searchCached: result.searchCached,
      terminated: result.terminated,
      depth,
      // DoneEvent.subQuestions: "Absent or 0 on a quick search" — result.subQuestions is
      // only ever set by runDeepLoop, so this spreads in nothing at all for a quick run.
      ...(result.subQuestions ? { subQuestions: result.subQuestions } : {})
    };

    // loop.ts's grounding backstop (StreamingGrounder for a live turn, or the fact that
    // finishOnCap's text is built directly from `sources`) already stripped any citation the
    // model cited without earning it (AGENTS.md's grounding rule is upheld either way) — this
    // is just so it shows up in the logs as a real signal that the prompt didn't work that
    // turn, not silence.
    if (result.groundingIssues) {
      log.warn({ requestId, groundingIssues: result.groundingIssues, depth }, 'model cited an unresolved source number; stripped before sending');
    }

    // Step 3, part 2: record the assistant's side of the conversation, so the next follow-up
    // in this thread (or a GET /threads/:id) sees it.
    await (await db()).collection('messages').insertOne({
      _id: newMessageId(),
      threadId,
      userId,
      role: 'assistant',
      content: result.answerText,
      answerId: done.answerId,
      sources: result.sources,
      done,
      createdAt: new Date()
    } as any);

    sseWrite(res, 'done', done);
    res.end();

    // Step 5: the run log. Exactly the RunLog shape from the contract (db.ts) — `tokens` is
    // ONE total here (in + out combined), unlike `done.tokens`, which keeps them split.
    // Wrapped in its own try/catch: a disk problem writing this file is a real bug worth
    // logging, but it must never reach the client — the SSE response is already sent and
    // ended above, so there is nothing left to report it to.
    try {
      const runLog: RunLog = {
        tokens: result.tokensIn + result.tokensOut,
        wallClockSec: (Date.now() - start) / 1000,
        costUsd: done.costUsd,
        terminated: result.terminated,
        depth,
        toolCalls: result.toolCalls
      };
      writeFileSync(join(env.runsDir, `${requestId}.json`), JSON.stringify(runLog, null, 2));
    } catch (err) {
      log.error({ err, requestId }, 'failed to write run log');
    }

    // Observability (AGENTS.md): one pino line per answer, and one row in `requests` —
    // the row is what makes GET /stats and the deep daily cap queryable instead of grep-only.
    // `toolCalls` here is a COUNT (RequestDoc, db.ts), unlike RunLog's array above.
    log.info(
      {
        requestId,
        toolCalls: result.toolCalls.length,
        terminated: result.terminated,
        tokens: { in: result.tokensIn, out: result.tokensOut },
        costUsd: done.costUsd,
        searchCached: result.searchCached,
        ttftMs: done.ttftMs,
        latencyMs: done.latencyMs,
        depth
      },
      'answer'
    );
    try {
      await (await db()).collection('requests').insertOne({
        requestId,
        userId,
        route: 'POST /threads/:threadId/ask',
        status: 200,
        ms: done.latencyMs,
        tokensIn: result.tokensIn,
        tokensOut: result.tokensOut,
        costUsd: done.costUsd,
        toolCalls: result.toolCalls.length,
        terminated: result.terminated,
        depth,
        // Not part of RequestDoc's contract fields, but nothing forbids extra fields on the
        // actual Mongo document — GET /stats needs both to compute searchCacheHitRatePct
        // and ttftP95Ms without re-parsing pino's stdout lines.
        searchCached: result.searchCached,
        ttftMs: done.ttftMs,
        createdAt: new Date()
      });
    } catch (err) {
      log.error({ err, requestId }, 'failed to write request log');
    }
  } catch (err) {
    // Fail loud: a provider exception ends the stream as an error, never a fabricated
    // answer. The route itself also reports 502-shaped info to the client over SSE, since
    // headers are already sent and a normal res.status(502) is no longer possible here.
    const message = err instanceof Error ? err.message : String(err);
    log.error({ err, threadId: req.params.threadId }, 'ask failed');
    sseWrite(res, 'error', { status: 502, error: message });
    res.end();

    // Still counted — a failed deep run already spent the LLM calls the cap exists to
    // bound, and still counts toward /stats.requests (an attempt, not an answer).
    try {
      await (await db()).collection('requests').insertOne({
        requestId,
        userId,
        route: 'POST /threads/:threadId/ask',
        status: 502,
        ms: Date.now() - start,
        terminated: 'error',
        depth,
        createdAt: new Date()
      });
    } catch (logErr) {
      log.error({ err: logErr, requestId }, 'failed to write request log');
    }
  }
});

// ---------------------------------------------------------------- GET /memory (list, step 4)

app.get('/memory', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }
  const rows = await (await db())
    .collection('memories')
    .find({ userId })
    .sort({ createdAt: -1 })
    .toArray();
  const body: ListMemoryResponse = {
    memories: rows.map((m) => ({
      id: m._id as unknown as string,
      text: m.text as string,
      sourceThread: m.sourceThread as string | undefined,
      createdAt: new Date(m.createdAt).toISOString()
    }))
  };
  res.json(body);
});

// ---------------------------------------------------------------- DELETE /memory/:id (step 4)

app.delete('/memory/:memoryId', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }
  const { memoryId } = req.params;
  // Scoped by userId too, same reasoning as the thread-ownership check: deleting someone
  // else's memory by guessing its id should look exactly like it never existed.
  const result = await (await db()).collection('memories').deleteOne({ _id: memoryId, userId } as any);
  if (result.deletedCount === 0) {
    return res.status(404).json({ error: `no memory ${memoryId}`, status: 404 });
  }
  res.status(204).end();
});

// ---------------------------------------------------------------- POST /spaces (step 6, part 1)
// A Space is just a named folder documents get uploaded into — deliberately as simple as a
// thread was in step 1: create the row, hand back its id. The upload/parse/index machinery
// (step 6, parts 2-3) is what makes a Space actually do something.

app.post('/spaces', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }
  const parsed = CreateSpaceBody.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.message, status: 400 });
  }
  const spaceId = newId('spc');
  await (await db()).collection('spaces').insertOne({
    _id: spaceId,
    userId,
    name: parsed.data.name,
    createdAt: new Date()
  } as any);
  const body: CreateSpaceResponse = { spaceId, name: parsed.data.name };
  res.status(201).json(body);
});

// ---------------------------------------------------------------- GET /spaces (list, step 6)

app.get('/spaces', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }
  const rows = await (await db())
    .collection('spaces')
    .find({ userId })
    .sort({ createdAt: -1 })
    .toArray();
  const body: ListSpacesResponse = {
    spaces: rows.map((s) => ({
      spaceId: s._id as unknown as string,
      name: s.name as string,
      createdAt: new Date(s.createdAt).toISOString()
    }))
  };
  res.json(body);
});

// ---------------------------------------------------------------- POST /spaces/:id/documents (step 6, part 2)
// AGENTS.md: this route does ONLY the cheap part — save the raw bytes, write a "pending" row
// each for the document and the job, respond 202. It must NOT parse/chunk/embed inline; that
// is the worker's job (worker.ts, part 3). upload.single('file') runs first (see the multer
// setup above) and reads the multipart body BEFORE this handler even starts — req.file is
// already fully populated by the time we get here.

app.post('/spaces/:spaceId/documents', upload.single('file'), async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }

  const { spaceId } = req.params;
  // Same ownership check as threads/memory: an unknown OR someone-else's spaceId 404s.
  const space = await (await db()).collection('spaces').findOne({ _id: spaceId, userId } as any);
  if (!space) {
    return res.status(404).json({ error: `no space ${spaceId}`, status: 404 });
  }

  const file = req.file;
  if (!file) {
    return res.status(400).json({ error: 'multipart field "file" is required', status: 400 });
  }
  if (!(ACCEPTED_UPLOAD_TYPES as readonly string[]).includes(file.mimetype)) {
    return res.status(400).json({
      error: `unsupported file type "${file.mimetype}" — accepted: ${ACCEPTED_UPLOAD_TYPES.join(', ')}`,
      status: 400
    });
  }

  // The ONLY content-touching work this route does: copy the raw bytes into GridFS.
  // openUploadStream() hands back its stream's id synchronously (before the write finishes)
  // — we still await 'finish' before trusting the bytes are durably stored.
  const bucket = new GridFSBucket(await db(), { bucketName: GRIDFS_BUCKETS.uploads });
  const uploadStream = bucket.openUploadStream(file.originalname, { contentType: file.mimetype });
  await new Promise<void>((resolve, reject) => {
    uploadStream.once('finish', () => resolve());
    uploadStream.once('error', reject);
    Readable.from(file.buffer).pipe(uploadStream);
  });
  const fileId = uploadStream.id.toString();

  const docId = newId('doc');
  await (await db()).collection('documents').insertOne({
    _id: docId,
    spaceId,
    userId,
    title: file.originalname,
    mimeType: file.mimetype,
    bytes: file.size,
    status: 'pending',
    pct: 0,
    fileId,
    createdAt: new Date()
  } as any);

  // The jobs row is the worker's to-do list entry. `newId()` doesn't have a 'job' prefix
  // (jobs aren't referenced by users the way a thread or doc id is), so this is the same
  // id-generation logic inlined, just tagged job_ instead.
  const jobId = `job_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  await (await db()).collection('jobs').insertOne({
    _id: jobId,
    kind: 'index_document',
    status: 'pending',
    payload: { docId, spaceId, userId, fileId, mimeType: file.mimetype },
    userId,
    attempts: 0,
    createdAt: new Date()
  } as any);

  const body: UploadDocumentResponse = { docId, status: 'pending' };
  res.status(202).json(body);
});

// ---------------------------------------------------------------- GET /spaces/:id/documents (step 6)

app.get('/spaces/:spaceId/documents', async (req, res) => {
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res.status(401).json({ error: 'X-User-Id is required', status: 401 });
  }
  const { spaceId } = req.params;
  const space = await (await db()).collection('spaces').findOne({ _id: spaceId, userId } as any);
  if (!space) {
    return res.status(404).json({ error: `no space ${spaceId}`, status: 404 });
  }
  const rows = await (await db())
    .collection('documents')
    .find({ spaceId, userId })
    .sort({ createdAt: -1 })
    .toArray();
  const body: ListDocumentsResponse = {
    documents: rows.map((d) => ({
      docId: d._id as unknown as string,
      title: d.title as string,
      status: d.status as DocStatus,
      pct: d.pct as number,
      pages: d.pages as number | undefined,
      chunks: d.chunks as number | undefined,
      error: d.error as string | undefined
    }))
  };
  res.json(body);
});

// ---------------------------------------------------------------- everything else: 501

const notImplemented = (route: string) => (_req: express.Request, res: express.Response) => {
  res.status(501).json({ error: `not implemented yet: ${route}. Build it in backend/agent/src/.`, status: 501 });
};

// A Set, not a chain of `route.path === '…' || …`: every one of ROUTES' 13 entries currently
// has a path in this list, so the OR-chain version let TypeScript prove the loop body below
// was unreachable (route narrowed all the way to `never`) — true today, but that was masking
// the real signal (nothing left to 501) behind a type error instead of just letting the loop
// harmlessly do nothing once every route ships. Set membership doesn't narrow route's type at
// all, which is exactly what's wanted here.
const IMPLEMENTED_PATHS = new Set([
  '/health',
  '/stats',
  '/evals/report.json',
  '/threads',
  '/threads/:threadId',
  '/threads/:threadId/ask',
  '/memory',
  '/memory/:memoryId',
  '/spaces',
  '/spaces/:spaceId/documents'
]);

for (const route of ROUTES) {
  if (IMPLEMENTED_PATHS.has(route.path)) continue;
  const method = route.method.toLowerCase() as 'get' | 'post' | 'delete';
  app[method](route.path, notImplemented(`${route.method} ${route.path}`));
}

app.use((req, res) => res.status(404).json({ error: `no route ${req.method} ${req.path}`, status: 404 }));

app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  // A file over MAX_UPLOAD_BYTES throws here as a MulterError with this code — the one
  // error this route can produce that isn't a plain 502-worthy surprise.
  if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ error: `file exceeds ${MAX_UPLOAD_BYTES} bytes`, status: 413 });
  }
  log.error({ err }, 'agent error');
  res.status(502).json({ error: err.message, status: 502 });
});

// '::' (all interfaces, IPv4+IPv6): required so this is reachable over Fly's private 6PN
// network, which is IPv6-only — the gateway reaches this service directly at
// http://<app>.internal:PORT. (.internal doesn't auto-wake a stopped machine, unlike the
// proxy-routed .flycast alternative — but .flycast needs an explicit IP allocation we'd
// rather not add right now, so this app is instead kept always-on via fly.toml instead.)
app.listen(env.port, '::', () => {
  log.info(
    {
      port: env.port,
      model: env.llmModel,
      searchProvider: env.searchProvider,
      vectorStore: env.vectorBackend,
      caps: {
        quick: { toolCalls: env.maxToolCalls, wallClockSec: env.maxWallClockSec },
        deep: { toolCalls: env.maxToolCallsDeep, wallClockSec: env.maxWallClockSecDeep, dailyCap: env.deepDailyCap }
      }
    },
    'agent up — /threads and /threads/:id/ask (quick) are live; the rest still returns 501'
  );
});
