/* OpenStat Studio — AI assistant command line (Gemini API), MIT License.
   A request typed in words is sent to /api/assistant/run; the server has the language model write program commands,
   checks them, runs them and returns the session with the new result cards (applyDatasetPayload).
   While a request runs the line shows the current step; when it fails it shows why.
   The API key goes to the local server only (memory, not the browser storage). */
(function () {
  "use strict";
  const en = () => typeof APP !== "undefined" && APP.lang === "en";
  const T = (ko, eng) => (en() ? eng : ko);
  const esc = (s) => String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  const STAGE = {
    route:  ["필요한 분석 고르는 중", "Choosing the analyses"],
    code:   ["명령 작성 중", "Writing the commands"],
    check:  ["명령 점검 중 (분석 이름·옵션·변수)", "Checking the commands (names, options, variables)"],
    fix:    ["명령 고쳐 쓰는 중", "Rewriting the commands"],
    run:    ["분석 실행 중", "Running the analysis"],
    report: ["결과 해석 작성 중", "Writing the interpretation"],
    commit: ["결과 시트에 올리는 중", "Adding to the results"],
  };
  const REASON = {
    key:      ["API 키 문제", "API key problem"],
    quota:    ["사용 한도 초과", "Quota exceeded"],
    network:  ["연결 실패", "Connection failed"],
    busy:     ["Gemini 서버 응답 없음", "Gemini server not responding"],
    model:    ["모형 사용 불가", "Model unavailable"],
    data:     ["자료 없음", "No data"],
    running:  ["처리 중인 요청 있음", "A request is running"],
    empty:    ["요청 없음", "Empty request"],
    api:      ["Gemini API 오류", "Gemini API error"],
    internal: ["프로그램 오류", "Program error"],
  };
  let info = { ready: false, has_key: false, models: [], examples: { ko: [], en: [] } };
  let timer = null, busy = false, lastDetail = null;

  function css() {
    const st = document.createElement("style");
    st.textContent = `
#ai-panel { flex: 0 0 auto; background: #F3F7FB; border-bottom: 1px solid #CBD5E1; padding: 8px 12px 7px; font-size: 12px; }
#ai-panel .ai-row { display: flex; align-items: center; gap: 8px; }
#ai-panel .ai-label { flex: 0 0 auto; font-weight: 700; font-size: 12.5px; color: #1E3A5F; white-space: nowrap; }
#ai-panel .ai-label small { display: block; font-weight: 500; font-size: 10.5px; color: #64748B; }
#ai-panel input[type=text], #ai-panel input[type=password] { min-width: 0; height: 34px; padding: 0 11px; font-size: 13.5px; border: 1.5px solid #1E3A5F; border-radius: 6px; outline: none; background: #FFFFFF; color: #1F2937; }
#ai-input { flex: 1 1 auto; }
#ai-panel input:focus { border-color: #0D9488; box-shadow: 0 0 0 3px rgba(13,148,136,.15); }
#ai-panel button.ai-main { flex: 0 0 auto; height: 34px; padding: 0 16px; font-size: 13px; font-weight: 700; color: #FFFFFF; background: #1E3A5F; border: 0; border-radius: 6px; cursor: pointer; }
#ai-panel button.ai-main:disabled { background: #94A3B8; cursor: default; }
#ai-panel .ai-sub { display: flex; align-items: center; gap: 6px; margin-top: 6px; flex-wrap: wrap; color: #475569; }
#ai-panel .ai-chip { padding: 2px 9px; border: 1px solid #94A3B8; border-radius: 999px; background: #FFFFFF; color: #1E3A5F; font-size: 11.5px; cursor: pointer; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#ai-panel .ai-chip:hover { border-color: #0D9488; color: #0F766E; }
#ai-status { margin-top: 6px; padding: 6px 9px; border-radius: 6px; font-size: 12px; line-height: 1.5; display: none; }
#ai-status.show { display: block; }
#ai-status.work { background: #E0F2FE; color: #075985; border: 1px solid #7DD3FC; }
#ai-status.ok { background: #ECFDF5; color: #065F46; border: 1px solid #6EE7B7; }
#ai-status.err { background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5; }
#ai-status b { font-weight: 700; }
#ai-status .ai-spin { display: inline-block; width: 11px; height: 11px; margin-right: 7px; vertical-align: -1px; border: 2px solid #7DD3FC; border-top-color: #075985; border-radius: 50%; animation: ai-spin .8s linear infinite; }
@keyframes ai-spin { to { transform: rotate(360deg); } }
#ai-status .ai-steps { display: flex; gap: 4px; margin-top: 5px; flex-wrap: wrap; }
#ai-status .ai-step { padding: 1px 8px; border-radius: 999px; font-size: 11px; background: #FFFFFF; color: #64748B; border: 1px solid #BAE6FD; }
#ai-status .ai-step.now { background: #075985; color: #FFFFFF; border-color: #075985; font-weight: 700; }
#ai-status .ai-step.done { color: #075985; }
#ai-status details { margin-top: 5px; }
#ai-status summary { cursor: pointer; font-weight: 600; }
#ai-status pre { margin: 4px 0 0; padding: 6px 8px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; font-size: 11.5px; white-space: pre-wrap; word-break: break-word; color: #1F2937; max-height: 180px; overflow: auto; }
#ai-keybox { display: none; margin-top: 7px; padding: 8px 10px; background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; }
#ai-keybox.open { display: block; }
#ai-keybox p { margin: 6px 0 0; font-size: 11px; color: #64748B; line-height: 1.5; }
#ai-key { flex: 1 1 auto; }
#ai-model { height: 30px; min-width: 240px; padding: 0 8px; font-size: 12.5px; border: 1.5px solid #1E3A5F; border-radius: 6px; background: #FFFFFF; color: #1F2937; }`;
    document.head.appendChild(st);
  }

  async function api(url, method, body) {
    const r = await fetch(url, { method: method || "GET", headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
    return r.json();
  }

  function setStatus(cls, html) {
    const el = document.getElementById("ai-status");
    if (!el) return;
    el.className = cls ? "show " + cls : "";
    el.innerHTML = html || "";
  }

  function idle() {
    if (busy) return;
    if (!info.has_key) setStatus("err", T("<b>준비 안 됨</b> — Gemini API 키를 등록해야 함. 아래 「API 키」에서 입력", "<b>Not ready</b> — register a Gemini API key under “API key” below"));
    else setStatus("", "");
  }

  function showProgress(p) {
    const order = ["route", "code", "check", "run", "report"];
    const stage = p.stage === "fix" ? "code" : p.stage === "commit" ? "report" : p.stage;
    const at = order.indexOf(stage);
    const chips = order.map((s, i) => `<span class="ai-step ${i === at ? "now" : i < at ? "done" : ""}">${i + 1}. ${esc(T(STAGE[s][0], STAGE[s][1]))}</span>`).join("");
    const label = STAGE[p.stage] ? T(STAGE[p.stage][0], STAGE[p.stage][1]) : T("처리 중", "Working");
    const sec = Math.round(p.elapsed_sec || 0);
    let extra = "";
    if ((p.attempt || 0) > 1) extra += " · " + T(`${p.attempt}번째 시도`, `attempt ${p.attempt}`);
    if (p.last_problem) extra += `<div style="margin-top:4px">${T("고치는 이유", "Reason for the rewrite")}: ${esc(p.last_problem)}</div>`;
    setStatus("work", `<span class="ai-spin"></span><b>${T("분석 중", "Working")}</b> — ${esc(label)} · ${sec}${T("초 경과", " s")}${extra}<div class="ai-steps">${chips}</div>`);
  }

  function startPolling() {
    showProgress({ stage: "route", elapsed_sec: 0 });
    const t0 = Date.now();
    timer = setInterval(async () => {
      try {
        const p = await api("/api/assistant/progress");
        if (p && p.busy) showProgress(p);
        else showProgress({ stage: "route", elapsed_sec: (Date.now() - t0) / 1000 });
      } catch (e) { /* the run request itself reports a lost connection */ }
    }, 1000);
  }
  function stopPolling() { if (timer) { clearInterval(timer); timer = null; } }

  function stepsHtml(a) {
    const bad = (a.steps || []).filter((s) => !s.ok);
    if (!bad.length) return "";
    const body = bad.map((s, i) => `${i + 1}. ${s.stage === "check" ? T("실행 전 점검에서 걸림", "Stopped by the pre-run check") : T("실행 오류", "Run error")}\n${s.error}\n--- ${T("작성된 명령", "commands written")} ---\n${s.code}`).join("\n\n");
    return `<details><summary>${T("고쳐 쓴 과정 보기", "Show the rewrites")} (${bad.length})</summary><pre>${esc(body)}</pre></details>`;
  }

  function showResult(a) {
    lastDetail = a;
    const tail = ` · ${T("모형 호출", "model calls")} ${a.n_calls ?? "-"} · ${a.elapsed_sec ?? "-"}${T("초", " s")} · Gemini (${esc(a.model || "-")})`;
    const skipped = (a.fallbacks || []).length ? `<div>${T("한도·오류로 건너뛴 모형", "Models skipped (quota or error)")}: ${esc(a.fallbacks.map((f) => f[0]).join(", "))}</div>` : "";
    if (a.executed) {
      const fixed = (a.steps || []).filter((s) => !s.ok).length;
      setStatus("ok", `<b>${T("분석 완료", "Done")}</b> — ${esc((a.analyses || []).join(", ") || T("자료 처리", "data step"))}`
        + (a.changes_data ? " · " + T("자료가 변경됨(실행 취소 가능)", "data changed (undo available)") : "")
        + (fixed ? " · " + T(`명령 ${fixed}회 수정 후 실행`, `ran after ${fixed} rewrite(s)`) : "") + tail + skipped
        + ((a.footer || []).length ? `<div>${esc(a.footer.join(" / "))}</div>` : "") + stepsHtml(a));
    } else if (a.failure) {
      setStatus("err", `<b>${T("분석 실패", "Analysis failed")}</b> — ${T("명령을 실행하지 못함. 결과 시트는 바뀌지 않음", "the commands could not be run; the results are unchanged")}`
        + `<div><b>${T("원인", "Reason")}</b>: ${esc(a.failure)}</div>` + tail + skipped + stepsHtml(a));
    } else {
      setStatus("ok", `<b>${T("응답 완료", "Answered")}</b> — ${T("실행한 분석 없음(되물음 또는 안내). 결과 시트의 응답 카드 확인", "no analysis was run (a question back or guidance); see the answer card")}` + tail);
    }
  }

  function showError(r) {
    const head = REASON[r.reason] ? T(REASON[r.reason][0], REASON[r.reason][1]) : T("오류", "Error");
    setStatus("err", `<b>${T("분석 실패", "Analysis failed")}</b> — ${esc(head)}<div><b>${T("원인", "Reason")}</b>: ${esc(r.message || r.detail || T("알 수 없는 오류", "unknown error"))}</div>`);
    if (r.reason === "key") document.getElementById("ai-keybox").classList.add("open");
  }

  async function run() {
    const input = document.getElementById("ai-input"), btn = document.getElementById("ai-run");
    const q = input.value.trim();
    if (!q || busy) return;
    busy = true; btn.disabled = true; btn.textContent = T("분석 중…", "Working…");
    startPolling();
    try {
      const r = await api("/api/assistant/run", "POST", { question: q, lang: en() ? "en" : "ko" });
      stopPolling();
      if (r.status !== "success") { showError(r); return; }
      if (typeof switchRightView === "function" && typeof APP !== "undefined" && APP.rightView === "data") switchRightView("results");
      applyDatasetPayload(r);
      showResult(r.assistant || {});
      setTimeout(() => {
        const cards = document.querySelectorAll(".jmv-results-sheet");
        const n = ((r.assistant || {}).analyses || []).length + 1;
        const first = cards[Math.max(0, cards.length - n)];
        if (first) first.scrollIntoView({ block: "start", behavior: "smooth" });
      }, 200);
    } catch (e) {
      stopPolling();
      showError({ reason: "network", message: T("프로그램 서버와 연결이 끊김: ", "Lost the connection to the program server: ") + e.message });
    } finally {
      busy = false; btn.disabled = false; btn.textContent = T("실행", "Run");
    }
  }

  function showModels() {
    const sel = document.getElementById("ai-model"), note = document.getElementById("ai-model-note");
    if (!sel) return;
    const list = info.models || [];
    sel.innerHTML = "";
    const auto = document.createElement("option");
    auto.value = ""; auto.textContent = T("자동 선택 (한도 초과·오류 시 다음 모형)", "Automatic (next model on quota or error)");
    sel.appendChild(auto);
    list.forEach((m) => { const o = document.createElement("option"); o.value = m; o.textContent = m; sel.appendChild(o); });
    sel.value = info.selected || "";
    sel.disabled = !info.has_key;
    note.textContent = info.has_key ? (list.length ? T(`이 키로 쓸 수 있는 모형 ${list.length}개`, `${list.length} models available to this key`) : T("모형 목록을 받지 못함(자동 선택으로 동작)", "Model list not received (automatic choice)")) : T("키를 등록하면 목록이 나타남", "Register a key to see the list");
  }

  function build() {
    const pane = document.getElementById("results-pane");
    const bar = pane && pane.querySelector(".right-pane-subbar");
    if (!bar) return;
    const old = document.getElementById("ai-panel");
    const keep = old ? { q: document.getElementById("ai-input").value, open: document.getElementById("ai-keybox").classList.contains("open") } : { q: "", open: false };
    if (old) old.remove();
    const el = document.createElement("div");
    el.id = "ai-panel";
    el.setAttribute("data-noi18n", "1");
    el.addEventListener("click", (e) => e.stopPropagation());
    const ex = (info.examples || {})[en() ? "en" : "ko"] || [];
    el.innerHTML = `
      <div class="ai-row">
        <div class="ai-label">${T("AI 분석 지원", "AI assistant")}<small>${T("말로 요청 → 분석 실행·해석", "ask in words → run & interpret")}</small></div>
        <input id="ai-input" type="text" autocomplete="off" placeholder="${T("예: 용량별로 치아 길이 평균이 다른지 검정하고 사후검정도 해줘", "e.g. Test whether tooth length differs by dose, with post-hoc comparisons")}">
        <button id="ai-run" class="ai-main">${T("실행", "Run")}</button>
      </div>
      <div class="ai-sub"><span>${T("예시", "Examples")}</span>${ex.map((x, i) => `<button class="ai-chip" data-i="${i}" title="${esc(x)}">${esc(x)}</button>`).join("")}
        <button class="ai-chip" id="ai-key-toggle" style="margin-left:auto">${T("API 키 · 모형", "API key · model")}</button></div>
      <div id="ai-status"></div>
      <div id="ai-keybox">
        <div class="ai-row">
          <input id="ai-key" type="password" autocomplete="off" placeholder="${T("Gemini API 키", "Gemini API key")}">
          <button id="ai-key-save" class="ai-main">${T("등록", "Register")}</button>
          <button id="ai-key-clear" class="ai-chip">${T("키 지우기", "Remove key")}</button>
        </div>
        <div class="ai-row" style="margin-top:8px">
          <div class="ai-label" style="font-size:12px">${T("모형", "Model")}</div>
          <select id="ai-model"></select>
          <span id="ai-model-note" style="font-size:11px;color:#64748B"></span>
        </div>
        <p>${T("키는 이 컴퓨터에서 실행 중인 프로그램 서버의 메모리에만 보관되며 프로그램을 끄면 사라짐. Google 로 전송되는 것: 요청 문장, 변수 설명(이름·유형·요약 통계·범주 이름), 분석 결과표. 자료의 개별 행은 전송되지 않음. 키 발급: aistudio.google.com/app/apikey · 무료 등급은 모형별 하루 요청 수 제한이 있음(요청 1건에 모형 호출 3회 안팎).",
               "The key is kept only in the memory of the program server on this computer and is gone when the program closes. Sent to Google: the request, variable descriptions (names, types, summary statistics, category names) and the result tables. Rows of the dataset are not sent. Get a key: aistudio.google.com/app/apikey · the free tier limits requests per day for each model (one request takes about 3 model calls).")}</p>
      </div>`;
    bar.insertAdjacentElement("afterend", el);
    const $ = (s) => el.querySelector(s);
    $("#ai-input").value = keep.q;
    if (keep.open || !info.has_key) $("#ai-keybox").classList.add("open");
    showModels();
    if (lastDetail && !busy) showResult(lastDetail); else idle();
    el.querySelectorAll(".ai-chip[data-i]").forEach((c) => c.addEventListener("click", () => { $("#ai-input").value = ex[+c.dataset.i]; $("#ai-input").focus(); }));
    $("#ai-run").addEventListener("click", run);
    $("#ai-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) run(); });
    $("#ai-key-toggle").addEventListener("click", () => $("#ai-keybox").classList.toggle("open"));
    $("#ai-key-save").addEventListener("click", async () => {
      const k = $("#ai-key").value.trim();
      if (!k) return;
      setStatus("work", `<span class="ai-spin"></span>${T("키 확인 중", "Checking the key")}`);
      try {
        const r = await api("/api/assistant/key", "POST", { key: k });
        $("#ai-key").value = "";
        if (r.status !== "success") { showError(r); return; }
        info = Object.assign(info, r); lastDetail = null;
        $("#ai-keybox").classList.remove("open"); showModels();
        setStatus("ok", `<b>${T("키 등록됨", "Key registered")}</b> — ${T("요청을 입력하고 실행", "type a request and run")}`);
      } catch (e) { showError({ reason: "network", message: e.message }); }
    });
    $("#ai-key-clear").addEventListener("click", async () => {
      const r = await api("/api/assistant/key", "DELETE");
      info = Object.assign(info, r); lastDetail = null; showModels(); idle();
    });
    $("#ai-model").addEventListener("change", async () => {
      const r = await api("/api/assistant/model", "POST", { model: $("#ai-model").value });
      if (r.status !== "success") { showError(r); return; }
      info = Object.assign(info, r); showModels();
    });
  }

  async function init() {
    css();
    try { info = Object.assign(info, await api("/api/assistant/info")); } catch (e) { /* server without the assistant */ }
    build();
    try {            // the forms of this page are the option reference the model is given
      if (typeof ANALYSIS_SCHEMAS === "object") api("/api/assistant/schemas", "POST", { schemas: JSON.parse(JSON.stringify(ANALYSIS_SCHEMAS)) }).catch(() => {});
    } catch (e) { /* older page */ }
    if (typeof setLanguage === "function") {
      const orig = setLanguage;
      window.setLanguage = function (lang) { const out = orig.apply(this, arguments); build(); return out; };
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
