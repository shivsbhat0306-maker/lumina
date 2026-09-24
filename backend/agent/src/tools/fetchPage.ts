/**
 * The fetch_page tool. Downloads a URL and extracts its main readable text with
 * @mozilla/readability (the same library Firefox's reader mode uses) so the model gets
 * article text, not nav bars and ads. This is what turns a search hit into something
 * actually citable.
 */
import { JSDOM, VirtualConsole } from 'jsdom';
import { Readability } from '@mozilla/readability';

export interface FetchedPage {
  url: string;
  title: string;
  text: string;
}

// JSDOM's default VirtualConsole forwards its own internal parsing errors (e.g. "Could
// not parse CSS stylesheet" whenever a page uses syntax its bundled CSS parser doesn't
// support, such as @layer) straight to the real console — dumping the entire raw
// stylesheet text. Readability strips <style> tags anyway, so we don't need that noise,
// and under concurrent fetch_page calls those large synchronous console.error dumps were
// blocking Node's single event loop thread for long enough to visibly delay other
// in-flight requests' SSE streaming. Give JSDOM its own console that quietly drops
// jsdom-internal errors instead of forwarding them.
const silentVirtualConsole = new VirtualConsole();

export async function fetchPage(url: string): Promise<FetchedPage> {
  const res = await fetch(url, {
    redirect: 'follow',
    headers: { 'user-agent': 'Mozilla/5.0 (compatible; LuminaBot/0.1)' }
  });
  if (!res.ok) {
    throw new Error(`fetch_page failed: ${res.status} ${url}`);
  }

  const html = await res.text();
  const dom = new JSDOM(html, { url, virtualConsole: silentVirtualConsole });
  const article = new Readability(dom.window.document).parse();

  if (!article?.textContent?.trim()) {
    throw new Error(`fetch_page: could not extract readable content from ${url}`);
  }

  return {
    url,
    title: article.title?.trim() || url,
    text: article.textContent.trim()
  };
}
