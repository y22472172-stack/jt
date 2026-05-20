"""
Local visual QA interface for bridge metric OCR results.

Run:
    python projects/elevation_detection/scripts/bridge_metrics_qa_server.py

Open:
    http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bridge_metrics_qa import DEFAULT_RESULT_JSON, DEFAULT_STORE, ask, build_store


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>&#26725;&#26753;&#22270;&#32440;&#25351;&#26631;&#38382;&#31572;</title>
  <style>
    * { box-sizing: border-box; }
    body { margin:0; font-family: "Microsoft YaHei", Arial, sans-serif; background:#f5f7fb; color:#18202f; }
    header { height:64px; display:flex; align-items:center; justify-content:space-between; padding:0 22px; background:#fff; border-bottom:1px solid #dce2ec; position:sticky; top:0; z-index:2; }
    h1 { margin:0; font-size:20px; }
    main { max-width:1480px; margin:0 auto; padding:18px; display:grid; grid-template-columns:420px minmax(0,1fr); gap:18px; }
    section { background:#fff; border:1px solid #dce2ec; box-shadow:0 8px 22px rgba(20,32,48,.08); }
    .head { height:46px; display:flex; align-items:center; justify-content:space-between; padding:0 16px; border-bottom:1px solid #dce2ec; font-weight:700; }
    .content { padding:16px; }
    textarea { width:100%; min-height:96px; resize:vertical; border:1px solid #dce2ec; border-radius:6px; padding:12px; font:inherit; outline:none; }
    textarea:focus { border-color:#126b62; box-shadow:0 0 0 3px #e3f2ef; }
    button { height:34px; border:1px solid #dce2ec; background:#fff; border-radius:6px; padding:0 12px; cursor:pointer; font:inherit; }
    button:hover { border-color:#126b62; color:#126b62; }
    .primary { background:#126b62; border-color:#126b62; color:#fff; font-weight:700; }
    .primary:hover { color:#fff; filter:brightness(.96); }
    .row { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
    .metrics { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:14px; }
    .metric { border:1px solid #dce2ec; border-radius:6px; padding:10px; background:#fbfcfe; min-height:66px; }
    .metric strong { display:block; font-size:20px; margin-bottom:4px; }
    .small, .hint { color:#687386; font-size:12px; }
    .answer { border-left:4px solid #126b62; background:#e3f2ef; padding:14px; line-height:1.75; white-space:pre-wrap; }
    .empty { color:#687386; border:1px dashed #dce2ec; background:#fbfcfe; padding:22px; text-align:center; }
    .evidence { border:1px solid #dce2ec; border-radius:6px; padding:10px; margin-top:10px; overflow-wrap:anywhere; }
    mark { background:#fff2d8; color:#8a5700; padding:1px 5px; border-radius:4px; }
    .kv { display:grid; grid-template-columns:82px minmax(0,1fr); gap:5px 8px; color:#687386; font-size:12px; margin-top:8px; }
    .tabs { display:flex; gap:6px; }
    .tab.active { background:#126b62; border-color:#126b62; color:#fff; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { padding:9px 8px; border-bottom:1px solid #dce2ec; text-align:right; white-space:nowrap; }
    th:first-child, td:first-child { text-align:left; }
    th { background:#fbfcfe; color:#687386; position:sticky; top:0; }
    .table-wrap { border:1px solid #dce2ec; border-radius:6px; max-height:470px; overflow:auto; }
    .bad { color:#a33c3c; }
    .viewer { margin-top:12px; border:1px solid #dce2ec; border-radius:6px; overflow:hidden; background:#eef1f6; }
    .viewer img { display:block; width:100%; max-height:420px; object-fit:contain; }
    .raw-list { display:flex; flex-wrap:wrap; gap:8px; }
    .raw-item { border:1px solid #dce2ec; background:#fbfcfe; border-radius:6px; padding:7px 9px; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } header { height:auto; align-items:flex-start; flex-direction:column; gap:8px; padding:14px 16px; } .metrics { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>&#26725;&#26753;&#22270;&#32440;&#25351;&#26631;&#38382;&#31572;</h1>
    <div class="small" id="sourceName">&#21152;&#36733;&#20013;</div>
  </header>
  <main>
    <section>
      <div class="head">&#38382;&#31572;&#26816;&#32034;</div>
      <div class="content">
        <textarea id="question" placeholder="&#20363;&#22914;&#65306;7&#21495;&#22697;&#22697;&#39640;&#22810;&#23569;&#65292;&#22475;&#28145;&#22810;&#23569;&#65311;"></textarea>
        <div class="row">
          <button class="primary" id="askBtn">&#26597;&#35810;</button>
          <button id="reloadBtn">&#37325;&#26032;&#21152;&#36733;&#25968;&#25454;</button>
        </div>
        <div class="metrics">
          <div class="metric"><strong id="mPiers">-</strong><span class="small">&#26725;&#22697;&#25968;&#37327;</span></div>
          <div class="metric"><strong id="mSpans">-</strong><span class="small">&#24635;&#36328;&#25968;</span></div>
          <div class="metric"><strong id="mLength">-</strong><span class="small">&#24635;&#38271; m</span></div>
          <div class="metric"><strong id="mSpanLen">-</strong><span class="small">&#26631;&#20934;&#36328;&#24452; m</span></div>
        </div>
        <div class="viewer"><img id="annotatedImage" alt="annotated"></div>
        <p class="hint">&#25903;&#25345;&#33258;&#28982;&#35821;&#35328;&#38382;&#31572;&#12290;&#22914;&#37197;&#32622;&#22823;&#27169;&#22411; API key&#65292;&#21518;&#31471;&#20250;&#20351;&#29992;&#22823;&#27169;&#22411;&#22312;&#35777;&#25454;&#22522;&#30784;&#19978;&#32452;&#32455;&#22238;&#31572;&#12290;</p>
      </div>
    </section>
    <section>
      <div class="head">
        <span>&#32467;&#26524;</span>
        <div class="tabs">
          <button class="tab active" data-tab="answer">&#38382;&#31572;</button>
          <button class="tab" data-tab="table">&#26725;&#22697;&#34920;</button>
          <button class="tab" data-tab="raw">OCR&#21407;&#25991;</button>
        </div>
      </div>
      <div class="content">
        <div id="answerTab">
          <div id="answerBox" class="empty">&#36755;&#20837;&#38382;&#39064;&#21518;&#28857;&#20987;&#26597;&#35810;&#12290;</div>
          <div id="evidenceList"></div>
        </div>
        <div id="tableTab" style="display:none">
          <div class="table-wrap">
            <table>
              <thead><tr><th>&#22697;&#21495;</th><th>top</th><th>middle</th><th>bottom</th><th>&#22697;&#39640;</th><th>&#22475;&#28145;</th><th>&#32852;</th><th>&#36328;&#24452;m</th><th>&#29366;&#24577;</th></tr></thead>
              <tbody id="pierRows"></tbody>
            </table>
          </div>
        </div>
        <div id="rawTab" style="display:none"><div id="rawList" class="raw-list"></div></div>
      </div>
    </section>
  </main>
<script>
let store = null;
const $ = (id) => document.getElementById(id);
const fmt = (v) => v === null || v === undefined ? "-" : v;
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));

async function loadStore() {
  const res = await fetch("/api/store");
  store = await res.json();
  $("sourceName").textContent = store.bridge_name || "loaded";
  const s = store.summary || {};
  $("mPiers").textContent = fmt(s.pier_count);
  $("mSpans").textContent = fmt(s.span_count);
  $("mLength").textContent = fmt(s.total_length_m);
  $("mSpanLen").textContent = (s.span_lengths || []).map(v => Number(v) / 100).join(", ") || "-";
  $("annotatedImage").src = "/artifact/page_0019_pier_metrics.png";
  renderTable();
  renderRaw();
}

function renderTable() {
  const rows = (store.facts || []).filter(f => f.type === "pier_metrics").map(f => {
    const bad = String(f.status || "").includes("missing") ? "bad" : "";
    return `<tr><td>${esc(f.pier_index)}号墩</td><td>${fmt(f.top_elevation)}</td><td>${fmt(f.middle_elevation)}</td><td>${fmt(f.bottom_elevation)}</td><td>${fmt(f.pier_height)}</td><td>${fmt(f.embed_depth)}</td><td>${fmt(f.span_group)}</td><td>${fmt(f.span_length_m)}</td><td class="${bad}">${esc(f.status || "")}</td></tr>`;
  }).join("");
  $("pierRows").innerHTML = rows;
}

function renderRaw() {
  $("rawList").innerHTML = (store.raw_ocr_docs || []).slice(0, 280).map(d => `<span class="raw-item" title="confidence=${esc(d.confidence)}">${esc(d.text)}</span>`).join("");
}

function renderEvidence(items, rawTexts) {
  if (!items.length) {
    $("evidenceList").innerHTML = `<div class="evidence"><span class="small">无证据链</span></div>`;
    return;
  }
  const evidence = items.map((item, i) => `<div class="evidence"><div><mark>${i + 1}</mark> ${esc(item.source_type || "source")} ｜ 原文：<strong>${esc(item.original_text || "")}</strong></div><div class="kv"><span>置信度</span><span>${fmt(item.confidence)}</span><span>数值</span><span>${fmt(item.value)}</span><span>bbox</span><span>${esc(JSON.stringify(item.bbox || []))}</span><span>来源</span><span>${esc(item.source || "")}</span></div></div>`).join("");
  const raw = rawTexts.length ? `<div class="evidence"><strong>召回原文</strong><div class="raw-list">${rawTexts.map(t => `<span class="raw-item">${esc(t)}</span>`).join("")}</div></div>` : "";
  $("evidenceList").innerHTML = evidence + raw;
}

async function askQuestion() {
  const question = $("question").value.trim();
  if (!question) return;
  $("answerBox").className = "answer";
  $("answerBox").textContent = "查询中...";
  $("evidenceList").innerHTML = "";
  const res = await fetch("/api/ask", {method:"POST", headers:{"Content-Type":"application/json; charset=utf-8"}, body:JSON.stringify({question})});
  const result = await res.json();
  $("answerBox").textContent = result.answer || "无答案";
  renderEvidence(result.evidence_chain || [], result.original_text || []);
}

document.querySelectorAll("[data-q]").forEach(btn => btn.addEventListener("click", () => { $("question").value = btn.dataset.q; askQuestion(); }));
document.querySelectorAll("[data-tab]").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  const tab = btn.dataset.tab;
  $("answerTab").style.display = tab === "answer" ? "" : "none";
  $("tableTab").style.display = tab === "table" ? "" : "none";
  $("rawTab").style.display = tab === "raw" ? "" : "none";
}));
$("askBtn").addEventListener("click", askQuestion);
$("reloadBtn").addEventListener("click", async () => { await fetch("/api/rebuild", {method:"POST"}); await loadStore(); $("answerBox").className = "empty"; $("answerBox").textContent = "数据已重新加载。"; $("evidenceList").innerHTML = ""; });
$("question").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) askQuestion(); });
loadStore().catch(err => { $("answerBox").className = "empty"; $("answerBox").textContent = err.message; });
</script>
</body>
</html>
"""


class BridgeQAHandler(BaseHTTPRequestHandler):
    store_path: Path = DEFAULT_STORE
    result_path: Path = DEFAULT_RESULT_JSON
    artifact_dir: Path = DEFAULT_STORE.parent

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def read_store(self) -> dict:
        if not self.store_path.exists():
            build_store(self.result_path, self.store_path)
        return json.loads(self.store_path.read_text(encoding="utf-8"))

    def send_json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_html(self) -> None:
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html()
            return
        if parsed.path == "/api/store":
            self.send_json(self.read_store())
            return
        if parsed.path.startswith("/artifact/"):
            name = Path(unquote(parsed.path.removeprefix("/artifact/"))).name
            path = self.artifact_dir / name
            if not path.exists() or not path.is_file():
                self.send_error(404, "artifact not found")
                return
            content = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/rebuild":
            store = build_store(self.result_path, self.store_path)
            self.send_json({"ok": True, "summary": store.get("summary")})
            return
        if parsed.path == "/api/ask":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self.send_json({"error": "invalid json"}, status=400)
                return
            question = str(payload.get("question") or "")
            if not question:
                self.send_json({"error": "question is required"}, status=400)
                return
            self.send_json(ask(self.read_store(), question))
            return
        self.send_error(404, "not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bridge metrics QA UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    BridgeQAHandler.store_path = args.store
    BridgeQAHandler.result_path = args.result_json
    BridgeQAHandler.artifact_dir = args.store.parent
    if not args.store.exists():
        build_store(args.result_json, args.store)
    server = ThreadingHTTPServer((args.host, args.port), BridgeQAHandler)
    print(f"Bridge metrics QA UI: http://{args.host}:{args.port}")
    print(f"Store: {args.store}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
