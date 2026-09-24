/**
 * LUMINA gateway — the software backend.
 *
 * What was already here: the server, CORS, the request id, the pino request log, /health
 * (which nests the agent service's health), and the static hosting of web/dist. That was
 * deliberately the boring half.
 *
 * STATUS: steps 1-5 done (README Part 2).
 *   1. X-User-Id enforcement (requireUser below) — 401 without it, on every route the
 *      contract marks auth:true (the exempt set is DERIVED from ROUTES, not hardcoded, so
 *      it can't quietly drift from the contract).
 *   2. zod validation from @lumina/contract (validateBody) — CreateThreadBody, AskBody,
 *      CreateSpaceBody — 400 with the zod message on a bad body. The multipart upload route
 *      has no JSON body to validate here; its file is validated agent-side, same as before.
 *   3. a per-user rate limit (rateLimit.ts, hand-rolled, no new dependency) — 429.
 *   4. the proxy: proxyJson handles every plain JSON route with ONE function (registered
 *      from ROUTES, so there's one code path, not one per route); POST /spaces/:id/documents
 *      forwards the raw multipart bytes untouched; POST /threads/:id/ask is the real
 *      streaming proxy — relays the agent's SSE bytes as they arrive, unbuffered (buffering
 *      here would quietly kill the TTFT SLA), but still passes a plain JSON error through
 *      verbatim when the agent rejected the request before ever starting a stream.
 *   5. GET /evals/report.json — serves reports/report.json off disk (Part 4; a 404 with a
 *      clear message until the eval has actually been run).
 * Every path here ends in whatever status the agent itself sent, or a 502 if the agent
 * couldn't be reached at all — never a 2xx papering over a failure.
 *
 * The browser talks ONLY to this service. No provider key is ever read here.
 */
import express from 'express';
import cors from 'cors';
import { pinoHttp } from 'pino-http';
import pino from 'pino';
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import type { ZodTypeAny } from 'zod';
import {
  HealthResponse,
  REQUEST_HEADER,
  ROUTES,
  USER_HEADER,
  CreateThreadBody,
  AskBody,
  CreateSpaceBody,
  MAX_UPLOAD_BYTES
} from '@lumina/contract';
import { env } from './env.js';
import { sseHeaders, sseSend } from './sse.js';
import { checkRateLimit } from './rateLimit.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use(cors({ origin: env.corsOrigins, credentials: false, exposedHeaders: [REQUEST_HEADER] }));

// One request id, reused if the caller sent one, generated if not, forwarded to the agent
// service and logged by both. This is what makes one request greppable end to end.
app.use((req, res, next) => {
  const id = (req.header(REQUEST_HEADER) ?? `req_${randomUUID().slice(0, 12)}`).trim();
  res.locals.requestId = id;
  res.setHeader(REQUEST_HEADER, id);
  next();
});

app.use(
  pinoHttp({
    logger: log,
    genReqId: (_req, res) => String(res.locals.requestId),
    customProps: (req, res) => ({
      requestId: res.locals.requestId,
      userId: req.header(USER_HEADER) ?? null
    }),
    // The ask route is a stream; one line when it closes is the useful line.
    autoLogging: true
  })
);

// JSON everywhere except the multipart upload route, which your handler owns.
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

// ---------------------------------------------------------------- step 1: X-User-Id (401)
// Derived from ROUTES' own `auth` flag instead of hardcoding "/health" here — if the
// contract ever adds another public route, this exemption follows it automatically instead
// of silently 401-ing something that was supposed to be open.
// Explicit `Set<string>`: without it, TS 5.6's inferred-type-predicate narrowing on the
// `.filter((r) => r.auth === false)` above types this Set to exactly the two auth:false path
// literals in ROUTES today — correct right now, but then `req.path` (always a plain string
// at runtime) can never satisfy `.has()`'s parameter type below. Widening here is the actual
// intent: this set's job is runtime membership-checking against an arbitrary request path,
// not staying pinned to whatever ROUTES happens to contain at compile time.
const NO_AUTH_PATHS: Set<string> = new Set(ROUTES.filter((r) => r.auth === false).map((r) => r.path));

function requireUser(req: express.Request, res: express.Response, next: express.NextFunction) {
  if (NO_AUTH_PATHS.has(req.path)) return next();
  const userId = req.header(USER_HEADER);
  if (!userId) {
    return res
      .status(401)
      .json({ error: `${USER_HEADER} is required`, status: 401, requestId: String(res.locals.requestId) });
  }
  next();
}
app.use(requireUser);

// ---------------------------------------------------------------- step 3: per-user rate limit (429)
// Runs AFTER requireUser on purpose: by this point req.header(USER_HEADER) is guaranteed to
// be present (or requireUser already sent the 401 and this never runs for that request), so
// there's always something real to key the limit on.
app.use((req, res, next) => {
  if (NO_AUTH_PATHS.has(req.path)) return next();
  const userId = req.header(USER_HEADER)!;
  if (!checkRateLimit(userId, env.rateLimitPerMinute)) {
    return res.status(429).json({
      error: `rate limit exceeded (${env.rateLimitPerMinute} requests/minute)`,
      status: 429,
      requestId: String(res.locals.requestId)
    });
  }
  next();
});

// ---------------------------------------------------------------- /health (implemented)

app.get('/health', async (_req, res) => {
  let ai: { status: 'ok' | 'down' } & Record<string, unknown> = { status: 'down' };
  try {
    const upstream = await fetch(`${env.agentUrl}/health`, { signal: AbortSignal.timeout(3000) });
    const body = (await upstream.json()) as Record<string, unknown>;
    ai = { ...body, status: upstream.ok ? 'ok' : 'down' };
  } catch (err) {
    // Health tells the truth about a dead dependency. It never pretends.
    ai = { status: 'down', error: (err as Error).message };
  }

  const body: HealthResponse = {
    status: ai.status === 'ok' ? 'ok' : 'degraded',
    model: String(ai.model ?? 'unset'),
    searchProvider: (ai.searchProvider as HealthResponse['searchProvider']) ?? 'tavily',
    vectorStore: (ai.vectorStore as HealthResponse['vectorStore']) ?? 'atlas-vector-search',
    db: (ai.db as HealthResponse['db']) ?? 'down',
    ai
  };
  res.status(ai.status === 'ok' ? 200 : 503).json(body);
});

// ---------------------------------------------------------------- GET /evals/report.json (step 5)
// Purely a static-file concern, same as webDist below — the gateway owns deploy artifacts,
// the agent service has no reason to know eval reports exist. Written by
// eval/build-report.mjs (Part 4), so a 404 here just means "haven't run the eval yet."

app.get('/evals/report.json', (_req, res) => {
  if (!existsSync(env.reportsFile)) {
    return res.status(404).json({
      error: 'no report yet — run the eval (TECHNICAL.md Part 4) to generate reports/report.json',
      status: 404
    });
  }
  res.setHeader('Content-Type', 'application/json');
  res.sendFile(env.reportsFile);
});

// ---------------------------------------------------------------- step 2: zod body validation (400)
// Only the two POST routes that go through the generic proxy loop below AND take a JSON
// body. AskBody validates POST /threads/:id/ask too, but that route is wired by hand further
// down (it needs the streaming proxy, not proxyJson) with validateBody(AskBody) applied
// directly there — it's deliberately not repeated in this map. POST /spaces/:id/documents is
// multipart, not JSON, so it has no entry here at all; see the raw upload proxy below.
const BODY_SCHEMAS: Partial<Record<string, ZodTypeAny>> = {
  'POST /threads': CreateThreadBody,
  'POST /spaces': CreateSpaceBody
};

function validateBody(schema: ZodTypeAny) {
  return (req: express.Request, res: express.Response, next: express.NextFunction) => {
    const parsed = schema.safeParse(req.body);
    if (!parsed.success) {
      return res
        .status(400)
        .json({ error: parsed.error.message, status: 400, requestId: String(res.locals.requestId) });
    }
    req.body = parsed.data; // normalized (defaults applied) before it's forwarded on
    next();
  };
}

// ---------------------------------------------------------------- step 4: the proxy

/**
 * Forwards a plain JSON (or bodyless) request to the agent service and relays its status
 * and body back verbatim. req.originalUrl already has every :param substituted with the
 * real value express matched (e.g. "/threads/thr_123"), and the agent's routes are the
 * SAME shape as the gateway's — so there's no path template to fill in, just forward it.
 * Used for every contract route except the streaming ask route and the raw upload route,
 * which each need different handling below.
 */
async function proxyJson(req: express.Request, res: express.Response): Promise<void> {
  const url = `${env.agentUrl}${req.originalUrl}`;
  const hasBody = req.method !== 'GET' && req.method !== 'DELETE';
  try {
    const upstream = await fetch(url, {
      method: req.method,
      headers: {
        [USER_HEADER]: req.header(USER_HEADER) ?? '',
        [REQUEST_HEADER]: String(res.locals.requestId),
        ...(hasBody ? { 'content-type': 'application/json' } : {})
      },
      body: hasBody ? JSON.stringify(req.body ?? {}) : undefined
    });
    const text = await upstream.text();
    res.status(upstream.status);
    res.setHeader('Content-Type', upstream.headers.get('content-type') ?? 'application/json');
    res.send(text);
  } catch (err) {
    // step 4/5: the agent couldn't be reached at all (down, DNS, connection refused) — a 502,
    // never a 2xx (rule A1: fail loud, never a try/catch that returns a plausible answer).
    log.error({ err, requestId: res.locals.requestId, url }, 'proxy to agent failed');
    res.status(502).json({ error: (err as Error).message, status: 502, requestId: String(res.locals.requestId) });
  }
}

// Register every contract route through proxyJson EXCEPT the two that need special
// handling (ask streams, documents is raw multipart) and the two static ones above. One
// loop, one code path — not hand-written per route.
const SPECIAL_ROUTES = new Set(['POST /threads/:threadId/ask', 'POST /spaces/:spaceId/documents']);
for (const route of ROUTES) {
  if (route.path === '/health' || route.path === '/evals/report.json') continue;
  if (SPECIAL_ROUTES.has(`${route.method} ${route.path}`)) continue;

  const method = route.method.toLowerCase() as 'get' | 'post' | 'delete';
  const schema = BODY_SCHEMAS[`${route.method} ${route.path}`];
  if (schema) {
    app[method](route.path, validateBody(schema), proxyJson);
  } else {
    app[method](route.path, proxyJson);
  }
}

// ---------------------------------------------------------------- step 4b: raw upload proxy
// The gateway never touches file content, same as it never touches provider keys — it just
// forwards the exact bytes and the exact Content-Type (multipart boundary and all) through.
// express.raw({ type: () => true }) parses regardless of the incoming content-type, ONLY on
// this one route (registered here, not as a global app.use, so nothing else is affected).

app.post(
  '/spaces/:spaceId/documents',
  express.raw({ type: () => true, limit: MAX_UPLOAD_BYTES }),
  async (req, res) => {
    const url = `${env.agentUrl}${req.originalUrl}`;
    try {
      const upstream = await fetch(url, {
        method: 'POST',
        headers: {
          [USER_HEADER]: req.header(USER_HEADER) ?? '',
          [REQUEST_HEADER]: String(res.locals.requestId),
          'content-type': req.header('content-type') ?? 'application/octet-stream'
        },
        body: req.body as Buffer
      });
      const text = await upstream.text();
      res.status(upstream.status);
      res.setHeader('Content-Type', upstream.headers.get('content-type') ?? 'application/json');
      res.send(text);
    } catch (err) {
      log.error({ err, requestId: res.locals.requestId, url }, 'proxy upload to agent failed');
      res.status(502).json({ error: (err as Error).message, status: 502, requestId: String(res.locals.requestId) });
    }
  }
);

// ---------------------------------------------------------------- step 4c: SSE streaming proxy
// POST /threads/:id/ask is the one route that can't go through proxyJson: the agent's
// response IS the product (trace/sources/token/done arriving live), so this has to relay
// bytes as they arrive rather than waiting for the whole response like proxyJson does.

app.post('/threads/:threadId/ask', validateBody(AskBody), async (req, res) => {
  const url = `${env.agentUrl}${req.originalUrl}`;

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers: {
        [USER_HEADER]: req.header(USER_HEADER) ?? '',
        [REQUEST_HEADER]: String(res.locals.requestId),
        'content-type': 'application/json'
      },
      body: JSON.stringify(req.body ?? {})
    });
  } catch (err) {
    log.error({ err, requestId: res.locals.requestId, url }, 'ask proxy: could not reach agent');
    res.status(502).json({ error: (err as Error).message, status: 502, requestId: String(res.locals.requestId) });
    return;
  }

  const isStream = (upstream.headers.get('content-type') ?? '').includes('text/event-stream');
  if (!isStream) {
    // The agent rejected the request before ever starting a stream — its OWN validation
    // (unknown threadId → 404, deep cap → 429, a bad spaceId → 404, etc.) sent a plain JSON
    // body instead. Relay it verbatim; there's nothing to stream.
    const text = await upstream.text();
    res.status(upstream.status);
    res.setHeader('Content-Type', upstream.headers.get('content-type') ?? 'application/json');
    res.send(text);
    return;
  }

  // From here the agent has already committed to a stream (its own headers are sent and
  // flushed on ITS side). Relay bytes as they arrive, unbuffered — this is the actual TTFT
  // SLA, so no reading the whole thing into memory first.
  sseHeaders(res);
  if (!upstream.body) {
    sseSend(res, 'error', { status: 502, error: 'agent returned an empty stream' });
    res.end();
    return;
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(decoder.decode(value, { stream: true }));
      // @ts-expect-error `flush` exists when a compression middleware is present; harmless otherwise.
      if (typeof res.flush === 'function') res.flush();
    }
  } catch (err) {
    // Too late to change the status code — headers are long gone. The client just sees the
    // stream end early, same as any dropped connection. Still logged, still greppable.
    log.error({ err, requestId: res.locals.requestId }, 'ask proxy: stream broke mid-flight');
  } finally {
    res.end();
  }
});

// ---------------------------------------------------------------- static UI

// In production the gateway serves the built UI, so / and /evals come from one origin.
if (existsSync(env.webDist)) {
  app.use(express.static(env.webDist));
  app.get(/^(?!\/(health|stats|threads|memory|spaces|artifacts|evals)).*/, (_req, res) => {
    res.sendFile(`${env.webDist}/index.html`);
  });
}

app.use((req, res) => {
  res.status(404).json({ error: `no route ${req.method} ${req.path}`, status: 404 });
});

// A thrown error is a 502 with a log line, never a 200 with a plausible body (rule A1).
// One special case: express.raw's own body-size guard on the upload route throws BEFORE our
// handler ever runs, as a "entity.too.large" error — that's a 413, not a generic 502.
app.use((err: any, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  if (err?.type === 'entity.too.large' || err?.status === 413) {
    res
      .status(413)
      .json({ error: `file exceeds ${MAX_UPLOAD_BYTES} bytes`, status: 413, requestId: String(res.locals.requestId) });
    return;
  }
  log.error({ err, requestId: res.locals.requestId }, 'gateway error');
  res.status(502).json({ error: err.message, status: 502, requestId: String(res.locals.requestId) });
});

app.listen(env.port, () => {
  log.info(
    {
      port: env.port,
      agentUrl: env.agentUrl,
      cors: env.corsOrigins,
      rateLimitPerMinute: env.rateLimitPerMinute
    },
    'gateway up — proxying every contract route to the agent service'
  );
});
