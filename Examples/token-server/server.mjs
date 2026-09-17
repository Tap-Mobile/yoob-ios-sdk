// Minimal backend for the QuickStart app. It keeps your Yoob API key on the server and hands the app a session.
//
//   YOOB_API_KEY=yoob_live_... node server.mjs
//
// In your own backend, check who the user is before opening a session: every session is metered to your workspace.
import http from "node:http";

const apiKey = process.env.YOOB_API_KEY;
const apiBase = process.env.YOOB_API_BASE ?? "https://api.yoob.com";
const port = Number(process.env.PORT ?? 3100);
if (!apiKey) throw new Error("Set YOOB_API_KEY");

http.createServer(async (req, res) => {
  if (req.method !== "POST" || req.url !== "/yoob-session") {
    res.writeHead(404).end();
    return;
  }
  let body = "";
  for await (const part of req) body += part;
  const { character } = JSON.parse(body || "{}");

  const upstream = await fetch(`${apiBase}/api/v1/avatar/sessions`, {
    method: "POST",
    headers: { authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
    body: JSON.stringify({ characters: [character] }),
  });
  res.writeHead(upstream.status, { "content-type": "application/json" });
  res.end(await upstream.text());
}).listen(port, () => console.log(`Yoob token server on http://localhost:${port}/yoob-session`));
