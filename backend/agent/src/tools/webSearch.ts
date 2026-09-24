/**
 * The web_search tool. Returns short {title, url, snippet} hits — never enough on its own
 * to cite from (AGENTS.md: "synthesize from fetched page text ... not from search
 * snippets"). Its only job is to give the loop candidate URLs to hand to fetch_page.
 *
 * SEARCH_PROVIDER is swappable via env with no code change (AGENTS.md). Only Tavily is
 * wired up here — add a SerpApi branch the same shape if you need it.
 */
import { env, secrets } from '../env.js';

export interface WebSearchHit {
  title: string;
  url: string;
  snippet: string;
}

export async function webSearch(query: string): Promise<WebSearchHit[]> {
  if (env.searchProvider !== 'tavily') {
    throw new Error(`search provider not implemented: ${env.searchProvider}`);
  }
  if (!secrets.tavily) {
    throw new Error('TAVILY_API_KEY is not set');
  }

  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      api_key: secrets.tavily,
      query,
      max_results: 5,
      include_answer: false
    })
  });

  if (!res.ok) {
    // Fail loud: let this throw all the way up. The loop turns it into a trace error and,
    // if it happens outside a tool call, the route turns it into a 502.
    throw new Error(`tavily search failed: ${res.status} ${await res.text()}`);
  }

  const data = (await res.json()) as {
    results: Array<{ title: string; url: string; content: string }>;
  };

  return data.results.map((r) => ({ title: r.title, url: r.url, snippet: r.content }));
}
