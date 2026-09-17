// Minimal backend for the Yoob examples. It keeps your keys on the server and hands the app short-lived credentials.
//
//   YOOB_API_KEY=yoob_live_... [OPENAI_API_KEY=sk-...] node server.mjs
//
// POST /yoob-session   → a Yoob session (character downloads and metering)
// POST /openai-secret  → an OpenAI Realtime client secret, for YoobConversation (only if OPENAI_API_KEY is set)
//
// In your own backend, check who the user is first: every session is metered to your workspace.
import http from "node:http";

const apiKey = process.env.YOOB_API_KEY;
const apiBase = process.env.YOOB_API_BASE ?? "https://api2.yoob.com";
const port = Number(process.env.PORT ?? 3100);
if (!apiKey) throw new Error("Set YOOB_API_KEY");

async function forward(res, upstream) {
  res.writeHead(upstream.status, { "content-type": "application/json" });
  res.end(await upstream.text());
}

http.createServer(async (req, res) => {
  let body = "";
  for await (const part of req) body += part;
  if (req.method === "POST" && req.url === "/yoob-session") {
    const { character } = JSON.parse(body || "{}");
    return forward(res, await fetch(`${apiBase}/api/v1/avatar/sessions`, {
      method: "POST",
      headers: { authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
      body: JSON.stringify({ characters: character ? [character] : ["*"] }),
    }));
  }
  if (req.method === "POST" && req.url === "/openai-secret") {
    if (!process.env.OPENAI_API_KEY) {
      res.writeHead(501, { "content-type": "application/json" });
      return res.end(JSON.stringify({ error: "Set OPENAI_API_KEY to enable conversations." }));
    }
    return forward(res, await fetch("https://api.openai.com/v1/realtime/client_secrets", {
      method: "POST",
      headers: { authorization: `Bearer ${process.env.OPENAI_API_KEY}`, "content-type": "application/json" },
      body: JSON.stringify({ session: { type: "realtime", model: "gpt-realtime" } }),
    }));
  }
  res.writeHead(404).end();
}).listen(port, "127.0.0.1", () => console.log(`Yoob token server on http://localhost:${port}`));
