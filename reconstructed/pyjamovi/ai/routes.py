# -*- coding: utf-8 -*-
"""Web API of the AI assistant (natural-language command line of the web UI).

  GET    /api/assistant/info        readiness, selectable models, examples
  POST   /api/assistant/key         {key, model?}  register a Gemini API key (kept in server memory only) · DELETE removes it
  POST   /api/assistant/model       {model}        "" = automatic choice
  POST   /api/assistant/schemas     {schemas}      analysis forms of the open browser window (keeps the option reference current)
  GET    /api/assistant/progress    what the running request is doing (stage, attempt, elapsed seconds)
  POST   /api/assistant/run         {question, lang}

A request that ran is added to the session exactly like code typed into the command editor: result cards, data
steps (undoable) and the command record, where the request is written as a comment above the code that ran.
A request that could not run changes nothing and returns the reason.
"""
import os
import threading
import time
from typing import Any, Callable, Dict, List

from fastapi import Body
from fastapi.responses import JSONResponse

from . import knowledge as K
from .assistant import Assistant
from .gemini_client import KEY_ENV, MODEL_ENV, GeminiClient, GeminiError, list_models
from .note import NOTE_TYPE, build_note

EXAMPLES = {
    "ko": ["수치형 변수의 기술통계와 상관행렬을 보여줘", "집단 간 평균 차이를 검정하고 사후검정까지 해줘",
           "회귀분석을 하고 다중공선성과 잔차 진단도 확인해줘"],
    "en": ["Show descriptive statistics and a correlation matrix of the numeric variables",
           "Test the mean difference between groups, with post-hoc comparisons",
           "Run a regression and check collinearity and residual diagnostics"],
}


def explain(e: Exception) -> Dict[str, str]:
    """Failure -> (reason code, sentence for the user). The code lets the screen say what to do next."""
    if isinstance(e, GeminiError):
        s = e.status
        if s in (401, 403) or (s == 400 and "API key" in str(e)):
            return {"reason": "key", "message": f"API 키를 확인하지 못함 — 키가 틀렸거나 권한이 없음. ({e})"}
        if s == 429:
            return {"reason": "quota", "message": f"Gemini 사용 한도 초과 — 무료 등급은 모형별 하루 요청 수 제한이 있음. 잠시 뒤 다시 시도하거나 「API 키」에서 다른 모형 선택. ({e})"}
        if s == 0:
            return {"reason": "network", "message": f"Google 서버에 연결하지 못함 — 인터넷 연결·방화벽 확인. ({e})"}
        if s in (500, 503):
            return {"reason": "busy", "message": f"Gemini 서버가 일시적으로 응답하지 않음 — 잠시 뒤 다시 시도. ({e})"}
        if s == 404:
            return {"reason": "model", "message": f"선택한 모형을 쓸 수 없음 — 「API 키」에서 다른 모형 선택. ({e})"}
        return {"reason": "api", "message": str(e)}
    return {"reason": "internal", "message": f"처리 중 오류: {type(e).__name__}: {e}"[:500]}


class Backend:
    def __init__(self):
        self.key = os.environ.get(KEY_ENV, "")
        self.model = os.environ.get(MODEL_ENV) or None
        self.models: List[str] = []
        self.last_model = None
        self.lock = threading.Lock()
        self.history: List[Dict[str, str]] = []
        self.history_dataset = None
        self.progress: Dict[str, Any] = {"busy": False}
        self.client_factory: Callable[..., Any] = GeminiClient        # replaced in tests

    def status(self) -> Dict[str, Any]:
        missing = [] if self.key else ["Gemini API 키가 입력되지 않음"]
        return {"ready": not missing, "missing": missing, "has_key": bool(self.key), "selected": self.model or "",
                "model": self.model or self.last_model or "", "models": self.models, "busy": bool(self.progress.get("busy"))}

    def load_models(self) -> List[str]:
        if self.key and not self.models:
            try:
                self.models = list_models(self.key)
            except Exception:                             # noqa: BLE001
                self.models = []
        return self.models

    def set_key(self, key, model=None):
        key = (key or "").strip()
        if not key:
            raise GeminiError(400, "키가 비어 있음. API key 입력 필요")
        models = list_models(key)                         # also proves the key is valid
        self.key, self.models, self.last_model = key, models, None
        if model is not None:
            self.model = model or None
        if self.model and self.model not in models:
            self.model = None

    def set_model(self, model):
        model = (model or "").strip()
        if model and self.load_models() and model not in self.models:
            raise GeminiError(404, f"이 키로 쓸 수 없는 모형: {model}")
        self.model, self.last_model = (model or None), None

    def tick(self, stage: str, detail: str = "") -> None:
        p = self.progress
        if stage == "check":
            p["attempt"] = int(detail or 1)
        p.update(stage=stage, detail=detail if stage != "check" else p.get("detail", ""), updated=time.time())
        if stage == "fix":
            p["last_problem"] = detail


def register(app, STATE, execute_syntax, commit_syntax) -> Backend:
    backend = Backend()
    app.state.assistant = backend

    def err(info: Dict[str, str]):
        return JSONResponse({"status": "error", **info})

    @app.get("/api/assistant/info")
    def assistant_info():
        backend.load_models()
        return {**backend.status(), "examples": EXAMPLES, "analyses": len(K.analysis_ids())}

    @app.post("/api/assistant/key")
    def assistant_key(body: dict = Body(...)):
        try:
            backend.set_key(body.get("key"), body.get("model"))
        except Exception as e:                            # noqa: BLE001
            return err(explain(e))
        return {"status": "success", **backend.status()}

    @app.delete("/api/assistant/key")
    def assistant_key_clear():
        backend.key, backend.models, backend.last_model = "", [], None
        return {"status": "success", **backend.status()}

    @app.post("/api/assistant/model")
    def assistant_model(body: dict = Body(...)):
        try:
            backend.set_model(body.get("model"))
        except Exception as e:                            # noqa: BLE001
            return err(explain(e))
        return {"status": "success", **backend.status()}

    @app.post("/api/assistant/schemas")
    def assistant_schemas(body: dict = Body(...)):
        return {"status": "success", "analyses": K.set_live_schemas(body.get("schemas") or {})}

    @app.get("/api/assistant/progress")
    def assistant_progress():
        p = dict(backend.progress)
        if p.get("busy"):
            p["elapsed_sec"] = round(time.time() - p.get("started", time.time()), 1)
        return p

    @app.post("/api/assistant/run")
    def assistant_run(body: dict = Body(...)):
        q = " ".join(str(body.get("question") or "").split())
        lang = "en" if body.get("lang") == "en" else "ko"
        if not q:
            return err({"reason": "empty", "message": "요청이 비어 있음"})
        if not backend.key:
            return err({"reason": "key", "message": "Gemini API 키가 입력되지 않음 — 「API 키」에서 등록"})
        df = STATE.session.df
        if df is None or df.shape[1] == 0:
            return err({"reason": "data", "message": "열려 있는 자료가 없음 — 자료 파일을 먼저 불러옴"})
        if not backend.lock.acquire(blocking=False):
            return err({"reason": "running", "message": "앞선 요청을 처리하는 중 — 끝난 뒤 다시 요청"})
        try:
            backend.progress = {"busy": True, "question": q, "stage": "route", "detail": "", "attempt": 0, "started": time.time()}
            if backend.history_dataset != STATE.dataset_name:
                backend.history, backend.history_dataset = [], STATE.dataset_name
            STATE.log_sync()                              # the record must hold the data this request runs on
            prev_script = STATE.current_python_script()

            def run(code: str) -> Dict[str, Any]:
                try:
                    res = execute_syntax(code)
                except Exception as e:                    # noqa: BLE001 — the message goes back to the model
                    return {"ok": False, "error": f"{type(e).__name__}: {e}"}
                return {"ok": True, "analyses": res[0], "stdout": res[1], "raw": res}

            sess = STATE.session
            client = backend.client_factory(backend.key, backend.model)
            out = Assistant(client).ask(
                q, df, run, name=STATE.dataset_name, lang=lang, history=backend.history, progress=backend.tick,
                meta={"value_labels": getattr(sess, "value_labels", None), "variable_labels": getattr(sess, "variable_labels", None),
                      "measure_levels": getattr(sess, "measure_levels", None)},
                existing=[a.get("title") for a in STATE.analyses_store if a.get("analysis_type") != NOTE_TYPE])
            backend.last_model = out.get("model") or backend.last_model
            backend.tick("commit")
            if out["executed"]:
                commit_syntax("자연어 요청: " + q, out["code"], False, prev_script, out["exec"]["raw"])
            else:
                STATE.cmdlog.note("자연어 요청 (실행된 명령 없음)", q)
            note = build_note({"question": q, "answer": out["final"], "footer": out["footer"], "model": out.get("model"),
                               "executed": out["executed"], "failure": out.get("failure") or "", "code": out["code"], "lang": lang})
            aid = STATE.next_id
            STATE.next_id += 1
            payload = note.to_payload(analysis_id=aid)
            payload["analysis_type"] = NOTE_TYPE
            payload["params"] = note.options
            STATE.analyses_store.append(payload)
            STATE.analysis_objects[aid] = note
            STATE.session.history = list(STATE.analysis_objects.values())
            STATE.revision += 1
            backend.history = (backend.history + [{"question": q, "code": out["code"]}])[-5:]
            resp = STATE.dataset_payload()
            resp["assistant"] = {k: out.get(k) for k in ("executed", "failure", "final", "footer", "code", "model", "n_calls", "elapsed_sec",
                                                         "changes_data", "analyses", "routed", "fallbacks")}
            resp["assistant"]["steps"] = [{"stage": s["stage"], "ok": s["ok"], "error": s["error"][:600], "code": s["code"]} for s in out["steps"]]
            resp["assistant"]["noteId"] = aid
            return resp
        except Exception as e:                            # noqa: BLE001
            return err(explain(e))
        finally:
            backend.progress = {"busy": False, "last_question": q, "finished": time.time()}
            backend.lock.release()

    return backend
