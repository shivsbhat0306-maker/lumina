# DESIGN.md — LUMINA

## Components

> Frontend for the UI
> Gateway for Authentication and Validation
> Orchestrator/agent
> Classifier/router to distinguish between the qucik and deep/complex quesion or document or web search
> Planner to be used for the query decompisition 
> Web search tool for searching the web and to return snippets
> Embedding provider to turn text into a list of numbers
> LLM to compose the citations gathered from the webseacrh and the document search 
> Vectore store for storing document chunks
> Job collection where the document upload will be stored 
> Cache to store the history for the web search
> Sesssion history
> SSE or Websocket for streaming an answer


## Responsibilities

> Orchestrator/agent will be the one tha will hold the provider key
> Frondend and Gateway will be talking to the browser, they cannot hold a provider key , as the frontend is not secure and gateway is used for authentication and Validations.
> Gateway will check the requests-per-minute before a request is allowed to go through and will also be holding the streaming connection open and relaying on the chunks from  Orchestrator through to the Frontend, since the 
  Orchestrator itself isn't allowed to reach the browser directly
> Orchestrator tracks the token spend per session
> Classification runs inside the Orchestrator as the first step for every question. It is the only thing that decides quick-vs-deep and doc-vs-web. It never calls Tavily, never touches the vector store, and never writes any part 
  of the answer,it only returns a decision that the rest of the Orchestrator's code acts on. Also, after the planner decomposes (which will also be placed inside the Orchestrator) the query and sends it to the Classifier which is inside the Orchestrator. Planner will by itself not decide which question needs a web search or a document search.
> The Orchestrator's ingestion step only marks a job 'completed' in the job collection after it has confirmed every chunk was successfully written to the vector store; if any chunk fails, the job is marked 'failed' rather than	  
  'completed.' The vector store's actual contents that are the source of truth i.e., job status is just a record of what the ingestion step observed, not an independent claim.
> The Orchestrator writes a new entry to Session history after every completed answer i.e., the question, the answer, and its citations. The Orchestrator is also the only reader: when it needs conversational context (for 
  example,  to help the classifier judge quick-vs-deep), it reads Session history itself and passes the relevant context along i.e., the classifier never queries Session history directly.
> Search-result caching lives in [a MongoDB collection / an in-memory store] that the Orchestrator controls i.e., never in Tavily itself, since Tavily is an external API with no notion of your application's cache. The  	 
  Orchestrator checks this cache before calling Tavily, and writes to it after a successful call.
> The LLM never decides which tools to call — that decision is made entirely by the classifier before the LLM is invoked.
> Vector store resposibility is to answer whatever query is handed over to it by the orchestrator, it has no opinion of its own about relevance. So when the Orchestrator asks it for the closest matches, if that threshold lives in  
  the Orchestrator's query, the vectore sore will have to answer it.
> The Web search (i.e., Tavily) and Embedding provider (i.e., OpenAI) cannot decide when they are called.



## Communication

How does each pair of components talk, and why that way? HTTP, SSE, a Mongo collection
polled by a worker. Say what happens to an in-flight request when the thing on the other
end is down.

> The Frontend talks to the Gateway over plain HTTP for ordinary requests (submitting a question, uploading a document), and over SSE (or a WebSocket) for receiving a streamed answer. If the Gateway is unreachable, the Frontend shows a connection error and can offer a retry — this is the direction that actually matters, since the Frontend is the one waiting on a response.
> The Gateway does two things itself, entirely in-process, before anything reaches the Orchestrator: authentication/validation and the requests-per-minute check. Neither of these involves calling out anywhere, so if either fails, the Gateway rejects the request immediately and returns an error straight to the Frontend, the Orchestrator never even sees that request. Once a request passes both checks, the Gateway forwards it to the Orchestrator over HTTP, and for a streaming response, the Gateway holds the SSE/WebSocket connection open to the Frontend while relaying chunks it receives from the Orchestrator. If that connection to the Orchestrator breaks mid-stream, the Gateway should pass a clear "answer incomplete" signal through to the Frontend, rather than leaving the connection open with nothing coming through.
> Classifier and Planner both live inside the Orchestrator's own process, so their communication with the Orchestrator (and each other) is a plain function call, not a network protocol so there's no "request" that can fail in the network sense, only a normal code exception if something inside them errors. The sequence is: the Orchestrator calls the classifier on the raw question to get a quick-vs-deep decision; if deep, the Orchestrator calls the planner, which returns a list of sub-questions; each sub-question then goes back through the classifier's doc-vs-web decision. At every step, the classifier and planner only return decisions or sub-questions — they never place a call to Tavily, OpenAI, or the vector store themselves. The Orchestrator is the only one that acts on what they return.
> When the Orchestrator needs a web search, it calls Tavily over HTTP and gets back raw snippets and URLs — never Claude, only the Orchestrator. If that call times out or errors, one reasonable choice is graceful degradation: if the question could still be answered from an uploaded document, the Orchestrator proceeds with the document-search path and notes in the answer that live web results weren't available; if there's no document to fall back on, it returns an explicit error instead of a fabricated answer.
> When the Orchestrator needs an embedding — either for a document chunk at upload time, or for a question at query time — it calls OpenAI over HTTP with the text and gets back a vector. That's the entire interaction; OpenAI never touches MongoDB. Once the Orchestrator has the vector back, it separately writes that vector (with the chunk's text) into the vector store itself, over a normal database write. If OpenAI is unreachable during document upload, the ingestion step marks that job "failed" rather than "completed,". If OpenAI is unreachable while trying to embed an incoming question, the Orchestrator can't run a document search at all for that turn — it should tell the user document search is temporarily unavailable rather than silently skipping it.
> All of MongoDB — vector store, job collection, session history, cache — talks to the Orchestrator through direct database reads and writes, not HTTP. Because nearly everything in your system depends on Mongo being reachable, a Mongo outage is closer to a full outage than a single failed call — the Orchestrator should catch it and have the Gateway return a clear "service unavailable" to the Frontend rather than hanging on a request that will never complete.A failed write to Session history probably shouldn't block the user from getting their actual answer — that failure can be logged and swallowed quietly, while a failed write to the vector store during ingestion is much more serious and should surface as a failed job.
> Once the Orchestrator has assembled context — retrieved chunks, search results, or both — it calls Claude over a streaming connection and forwards the tokens back through the Gateway to the Frontend as they arrive. If that stream drops partway through, the Orchestrator should detect the break and send an explicit "answer incomplete" signal down the same chain, so the Frontend can show that clearly instead of leaving a half-written answer sitting there looking finished. 


## State


> Frontend, Gateway, and the Orchestrator itself hold no persistent state — the Gateway may keep short-lived, in-memory counters for rate-limiting, but nothing there survives a restart or needs to. Every piece of state that matters lives in MongoDB, in four collections, and the Orchestrator is the sole owner of writes to all of them.
> The vector store holds each document's chunked text alongside its embedding vector. 
> The job collection holds one status record per uploaded document, tracking its progress through the ingestion pipeline.
> Session history holds each session's past questions, answers, and citations. The search cache holds past web queries and their Tavily results with an expiry.
> 

## Trade-offs

> Hardcoded routing instead of letting the LLM decide which tools to call: the classifier and planner make every quick-vs-deep and doc-vs-web decision in plain code, before Claude ever sees the question, rather than giving Claude the tool definitions and letting it decide mid-conversation to search or look something up.
> Chunking and embedding a document synchronously, inside the same request that accepted the upload, instead of queuing it for a background worker: this avoids building any queue infrastructure at all, which is a real win for a first version. The cost is that a large document blocks the upload request for as long as embedding takes, with no natural retry if it fails partway through — a dropped connection mid-embedding just leaves a partially-processed document with no automatic recovery path.
> Not retaining the original uploaded file after it's chunked and embedded: only the processed chunks and their vectors persist in the vector store. This keeps the data model to one source of state instead of two, and saves on storage — but it means the vector store can never be cheaply rebuilt. If the chunking strategy or embedding model ever needs to change later, there's no original to re-process; the user would have to re-upload.
> One provider per role — OpenAI for embeddings, Anthropic for the LLM, Tavily for search — called directly, with no abstraction layer in front of any of them: this is by far the fastest way to ship a working version, since there's nothing to design beyond "call this API." The trade-off is vendor lock-in — if Tavily's results turned out to be worse than a competitor's, or pricing became a problem, swapping it out means touching every place in the Orchestrator that calls it, rather than changing one configuration value behind an interface.
