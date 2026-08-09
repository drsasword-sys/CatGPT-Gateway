# ChatGPT Project + lane routing

This branch adds an opt-in routing contract for clients that want one ChatGPT
Web Project per story and ordered conversations per lane.

## Contract

Send normal `POST /v1/chat/completions` requests with optional metadata:

```json
{
  "model": "catgpt-browser",
  "messages": [{"role": "user", "content": "..."}],
  "metadata": {
    "story_id": "story-1",
    "project_ref": "https://chatgpt.com/g/g-p-story/project",
    "chat_ref": "conversation-id-from-the-previous-response",
    "lane": 1,
    "lane_count": 10,
    "chapter": 11,
    "request_id": "idempotency-key-from-the-client"
  }
}
```

The response includes `metadata.thread_id`. The client must persist that value
as the lane's `chat_ref` and send it on the next request. A lane is serialized
by its own lock; different lanes may use different pages concurrently. The
gateway still keeps the legacy no-metadata endpoint on its original single-page
lock.

## Project creation

`POST /projects` drives the visible ChatGPT Web Project UI. It accepts a name,
optional instructions, and up to 25 base64-encoded reference files. The
endpoint fails with a warning/error if the UI selectors no longer expose a
Project control; it never silently falls back to an ordinary chat.

Project URLs are treated as opaque refs because ChatGPT has used more than one
URL shape. No browser cookies, local storage, or private backend API is read by
this feature.

## Resource and correctness limits

- Maximum configured lane count: 10.
- Same lane: strict chapter order and one in-flight browser operation.
- Different lanes: bounded by `max_concurrency` (default 10).
- A lane starts a fresh Project chat after eight messages, matching the
  existing gateway protection against long-thread UI degradation.
- The App-Stran client keeps a separate local request gate per lane and stores
  Project/lane/task state in SQLite for restart/resume.

Live Project UI creation and a real multi-lane UAT still require a logged-in
ChatGPT Web session; syntax/static tests do not prove those external selectors.
