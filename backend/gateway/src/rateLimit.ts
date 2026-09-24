/**
 * Step 3: a per-user rate limit. Hand-rolled, no new dependency — this is small enough not
 * to need one, same spirit as the agent's loop being "yours, not a framework's."
 *
 * Fixed 60-second window per X-User-Id: the first request in a window starts it, every
 * request within RATE_LIMIT_PER_MINUTE of it is allowed, the next one is rejected until the
 * window rolls over. This is a coarse abuse guardrail, not a precise SLA — a real token
 * bucket would smooth out a burst that lands right on a window boundary, but for a
 * bootcamp-scale gateway that precision isn't worth the extra code.
 */
interface Bucket {
  count: number;
  windowStart: number;
}

const WINDOW_MS = 60_000;
const buckets = new Map<string, Bucket>();

/** true = allowed (and counted against this window); false = over the limit. */
export function checkRateLimit(userId: string, limitPerMinute: number): boolean {
  const now = Date.now();
  const existing = buckets.get(userId);

  if (!existing || now - existing.windowStart >= WINDOW_MS) {
    buckets.set(userId, { count: 1, windowStart: now });
    return true;
  }

  if (existing.count >= limitPerMinute) {
    return false;
  }

  existing.count += 1;
  return true;
}

// Without this, a long-running process accumulates one Map entry per distinct userId
// forever. A coarse 10-minute sweep is plenty at bootcamp scale; .unref() so this timer
// alone never keeps the process alive (matters for clean shutdown / tests).
setInterval(() => {
  const cutoff = Date.now() - WINDOW_MS;
  for (const [userId, bucket] of buckets) {
    if (bucket.windowStart < cutoff) buckets.delete(userId);
  }
}, 10 * 60_000).unref();
