# -*- coding: utf-8 -*-
"""Gemini API client of the AI assistant (standard library only).

- The key travels only in the ``x-goog-api-key`` header; it is never written to a URL, file or log.
- Without a fixed model the models the key can use are listed and tried in order (flash family first);
  quota (429), retired model (404) and temporary failures (500/503) wait, retry, then move to the next model.
- ``chat(messages)`` takes OpenAI-style messages (system / user / assistant) and returns text;
  ``chat_json(messages, schema)`` asks for a JSON answer constrained by a response schema.

Sent to Google: the question, variable descriptions (names, types, summary statistics, category labels),
the option reference of the program and the result tables of the analyses. Rows of the dataset are not sent.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY_ENV = "GEMINI_API_KEY"
MODEL_ENV = "GEMINI_MODEL"
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]
_SKIP = ("preview", "exp", "tts", "image", "embedding", "live", "audio", "omni", "transcribe", "robotics",
         "computer", "banana", "research", "customtools")
_THINK = re.compile(r"<think>.*?</think>", re.S)


class GeminiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"Gemini API 오류 (HTTP {status}): {message}")
        self.status = status
        self.retry_after = None


def rank_model(name: str) -> int:
    n = name.lower()
    if not n.startswith("gemini") or any(w in n for w in _SKIP):
        return -1
    if "flash-lite" in n:
        return 80
    if "flash" in n:
        return 100
    if "pro" in n:
        return 40
    return 10


_RETRY_IN = re.compile(r"retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.I)
_BLOCKED = {}             # model -> time until which its quota is used up (shared by the clients of this process)


def retry_seconds(message):
    """Waiting time Google names in a quota error ("Please retry in 3h31m45.2s"), or None."""
    m = _RETRY_IN.search(message or "")
    if not m or not any(m.groups()):
        return None
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3) or 0)


def _request(url, key, body=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["x-goog-api-key"] = key
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 — fixed Google API address
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8"))["error"]["message"]
        except Exception:                                             # noqa: BLE001
            msg = e.reason
        err = GeminiError(e.code, str(msg)[:300])
        err.retry_after = retry_seconds(str(msg))            # waiting time named in a quota error (the text above is cut)
        raise err from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise GeminiError(0, f"연결하지 못함 ({getattr(e, 'reason', e)})"[:300]) from None


def list_models(key, timeout=20):
    d = _request(f"{BASE}/models?pageSize=200", key, timeout=timeout)
    names = [m["name"].replace("models/", "", 1) for m in d.get("models", [])
             if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    names = [n for n in names if rank_model(n) > 0]
    return sorted(names, key=lambda n: (-rank_model(n), [-ord(c) for c in n]))


def to_gemini(messages):
    """OpenAI-style messages -> (systemInstruction, contents); consecutive turns of one role are merged."""
    system, contents = [], []
    for m in messages:
        if m["role"] == "system":
            system.append(m["content"])
            continue
        role = "model" if m["role"] == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n\n" + m["content"]
        else:
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return ("\n\n".join(system) or None), contents


class GeminiClient:
    def __init__(self, api_key=None, model=None, temperature=0.0, max_tokens=3000, timeout=90, retries=3,
                 thinking_budget=0):
        self.key = api_key if api_key is not None else os.environ.get(KEY_ENV, "")
        self.model = model or None
        self.temperature, self.max_tokens, self.timeout, self.retries = temperature, max_tokens, timeout, retries
        self.thinking_budget = thinking_budget
        self.models = None
        self.no_thinking_cfg = set()
        self.used_model = None
        self.fallbacks = []       # (model, reason) of the models passed over in the last call
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def available(self):
        if self.models is None:
            try:
                self.models = list_models(self.key)[:8]
            except GeminiError as e:
                if e.status in (400, 401, 403):
                    raise
                self.models = []
            self.models = self.models or list(FALLBACK_MODELS)
        return self.models

    def candidates(self, light=False):
        """Models to try in order. A chosen model comes first and the rest stay as a fallback for a used-up quota;
        light=True (short classification calls) starts with the flash-lite family, whose quota is counted separately."""
        pool = list(self.available())
        if light:
            pool.sort(key=lambda n: 0 if "lite" in n else 1)
        elif self.model:
            pool = [self.model] + [m for m in pool if m != self.model]
        now = time.time()
        open_ = [m for m in pool if _BLOCKED.get(m, 0) <= now]
        return open_ or pool[:1]

    def _generate(self, model, system, contents, thinking, schema, max_tokens):
        cfg = {"temperature": self.temperature, "maxOutputTokens": max_tokens}
        if thinking and self.thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        else:
            cfg["maxOutputTokens"] = max_tokens * 3        # a model that always thinks: leave room for the thinking share
        if schema is not None:
            cfg["responseMimeType"] = "application/json"
            cfg["responseSchema"] = schema
        body = {"contents": contents, "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return _request(f"{BASE}/models/{model}:generateContent", self.key, body, self.timeout)

    def chat(self, messages, schema=None, max_tokens=None, light=False):
        if not self.key:
            raise GeminiError(401, "API 키가 없음")
        system, contents = to_gemini(messages)
        last = None
        self.fallbacks = []
        for model in self.candidates(light):
            thinking = model not in self.no_thinking_cfg
            for attempt in range(self.retries):
                try:
                    d = self._generate(model, system, contents, thinking, schema, max_tokens or self.max_tokens)
                except GeminiError as e:
                    last = e
                    if e.status == 400 and thinking and self.thinking_budget is not None:
                        thinking = False                    # the model rejects the thinking setting: call without it
                        self.no_thinking_cfg.add(model)
                        continue
                    if e.status == 429:
                        wait = getattr(e, "retry_after", None)
                        if wait is not None and wait > 40:  # daily quota used up: no point waiting, use the next model
                            _BLOCKED[model] = time.time() + wait
                            break
                        if attempt < self.retries - 1:
                            time.sleep(min((wait or 2 ** attempt * 3) + 1, 41))
                            continue
                        break
                    if e.status in (500, 503) and attempt < 1:
                        time.sleep(3)
                        continue
                    if e.status in (0, 404, 500, 503):
                        break
                    raise
                u = d.get("usageMetadata") or {}
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += u.get("promptTokenCount", 0)
                self.usage["completion_tokens"] += u.get("candidatesTokenCount", 0)
                cand = (d.get("candidates") or [{}])[0]
                text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
                if not text:
                    reason = cand.get("finishReason") or (d.get("promptFeedback") or {}).get("blockReason") or "빈 응답"
                    last = GeminiError(200, f"응답 본문 없음({reason})")
                    break
                if not light:
                    self.used_model = model
                return _THINK.sub("", text).strip()
            self.fallbacks.append((model, getattr(last, "status", None)))
        if last is not None and last.status == 429:
            raise GeminiError(429, "쓸 수 있는 모형의 사용 한도가 모두 소진됨(무료 등급은 모형별 하루 요청 수 제한이 있음). "
                                   "잠시 뒤 다시 시도하거나 다른 모형을 고름")
        raise last or GeminiError(0, "사용할 수 있는 모형이 없음")

    def chat_json(self, messages, schema, max_tokens=600, light=True):
        text = self.chat(messages, schema=schema, max_tokens=max_tokens, light=light)
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0) if m else text)
