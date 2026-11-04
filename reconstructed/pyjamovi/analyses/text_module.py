"""
OpenStat Text Mining Module (Korean-first)
- Morphological tokenization with kiwipiepy: part-of-speech filter, compound-noun joining,
  user dictionary, stopwords, synonym merging (tokenization results are cached per corpus)
- textKeywords : keyword frequency / document frequency / TF-IDF, group keywords (c-TF-IDF), word clouds
- textNetwork  : keyword co-occurrence network (sentence or document window), Louvain communities
- textCluster  : document embedding (TF-IDF + LSA by default, optional sentence-transformers such as
                 BAAI/bge-m3) -> UMAP -> HDBSCAN / k-means / Ward clusters, c-TF-IDF cluster keywords,
                 noise re-assignment, theme grouping, multi-seed stability (ARI), cluster map and
                 relationship networks

Clustering defaults follow the institute's research-field classification pipeline
(UMAP n_neighbors=12, min_dist=0, cosine; HDBSCAN eom; noise re-assignment at cosine >= 0.6;
themes by average-linkage cosine distance 0.3; stability seeds 42 / 7 / 2026).
"""

import hashlib
import bisect
import re
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from .base import BaseAnalysis, ResultsTable
from .plots import fig_to_base64, _pal
from .plot_style import get_active_style
from .base import resolve_choice
from .graph_network_module import (
    require_networkx, compute_centrality, detect_communities, community_modularity,
    plot_network_map, add_centrality_table, resolve_layout,
)

POS_MODES = {
    "nouns": {"NNG", "NNP"},
    "nouns_foreign": {"NNG", "NNP", "SL", "SH"},
    "nouns_predicates": {"NNG", "NNP", "SL", "SH", "VV", "VA"},
}
POS_LABELS = {
    "nouns": "명사",
    "nouns_foreign": "명사 + 영문·한자",
    "nouns_predicates": "명사 + 영문·한자 + 동사·형용사",
}

DEFAULT_STOPWORDS = set("""
것 수 등 및 때 중 년 월 일 위 이 그 저 곳 점 바 데 개 명 건 측 경우 정도 이상 이하 이후 이전 이내
대상 관련 내용 부분 사항 기타 해당 각 전 후 간 내 외 시 총 약 여 번 차 회 가지 우리 이번 이것 그것
자체 대한 통한 위한 따른 하다 되다 있다 없다 같다 이다 아니다 않다 보다 주다 받다 대하다 위하다 통하다
""".split())

STABILITY_SEEDS = (42, 7, 2026)

_KIWI_CACHE: "OrderedDict[Tuple[str, ...], Any]" = OrderedDict()
_TOKEN_CACHE: "OrderedDict[str, List[List[tuple]]]" = OrderedDict()
_EMBED_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_MODEL_CACHE: Dict[str, Any] = {}


def _cache_put(cache: OrderedDict, key, value, limit: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _get_kiwi(user_words: Tuple[str, ...]):
    if user_words in _KIWI_CACHE:
        _KIWI_CACHE.move_to_end(user_words)
        return _KIWI_CACHE[user_words]
    try:
        from kiwipiepy import Kiwi
    except ImportError as e:  # pragma: no cover - depends on the install profile
        raise ValueError("텍스트 분석에 kiwipiepy 패키지 필요: 'pip install kiwipiepy' 실행 후 다시 시도") from e
    kiwi = Kiwi()
    for w in user_words:
        kiwi.add_user_word(w, "NNP", 0)
    _cache_put(_KIWI_CACHE, user_words, kiwi, 4)
    return kiwi


def parse_word_list(value: Any) -> List[str]:
    """'a, b c\nd' -> ['a', 'b', 'c', 'd'] (lists are accepted as-is)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value]
    else:
        items = re.split(r"[,\s;]+", str(value))
    return [w.strip() for w in items if w and w.strip()]


def parse_synonyms(value: Any) -> Dict[str, str]:
    """
    One rule per line. 'target: source1, source2' or 'source1, source2 = target' (or '->').
    Dict input {source: target} is also accepted.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip() and str(v).strip()}
    out: Dict[str, str] = {}
    for line in str(value).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            target, sources = line.split(":", 1)
        elif "->" in line:
            sources, target = line.split("->", 1)
        elif "=" in line:
            sources, target = line.split("=", 1)
        else:
            continue
        target = target.strip()
        for s in parse_word_list(sources):
            if s and target and s != target:
                out[s] = target
    return out


class TextPrep:
    """Tokenization settings shared by every text analysis."""

    def __init__(self, pos_mode: str = "nouns_foreign", min_len: int = 2, compound: bool = True,
                 default_stop: bool = True, stopwords: Any = "", user_words: Any = "", synonyms: Any = "",
                 max_chars: int = 100000):
        self.pos_mode = resolve_choice("품사 범위", pos_mode, list(POS_MODES),
                                       {"noun": "nouns", "nouns_only": "nouns", "nouns_sl": "nouns_foreign",
                                        "nouns_verbs": "nouns_predicates", "predicates": "nouns_predicates"},
                                       default="nouns_foreign")
        self.min_len = max(int(min_len or 1), 1)
        self.compound = bool(compound)
        self.default_stop = bool(default_stop)
        self.user_stop = set(parse_word_list(stopwords))
        self.user_words = tuple(sorted(set(parse_word_list(user_words))))
        self.synonyms = parse_synonyms(synonyms)
        self.max_chars = int(max_chars or 100000)

    @property
    def stopwords(self) -> set:
        return (DEFAULT_STOPWORDS | self.user_stop) if self.default_stop else set(self.user_stop)

    def describe(self) -> str:
        parts = [f"품사: {POS_LABELS[self.pos_mode]}", f"최소 {self.min_len}글자"]
        if self.compound:
            parts.append("붙여 쓴 명사 결합")
        parts.append(f"불용어 {len(self.stopwords)}개" + (" (기본 목록 포함)" if self.default_stop else ""))
        if self.user_words:
            parts.append(f"사용자 사전 {len(self.user_words)}개")
        if self.synonyms:
            parts.append(f"유의어 규칙 {len(self.synonyms)}개")
        return ", ".join(parts)

    def raw_tokens(self, texts: List[str]) -> List[List[tuple]]:
        h = hashlib.sha1()
        h.update(("|".join(self.user_words) + f"|{self.max_chars}").encode("utf-8"))
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8", "ignore"))
        key = h.hexdigest()
        if key in _TOKEN_CACHE:
            _TOKEN_CACHE.move_to_end(key)
            return _TOKEN_CACHE[key]
        kiwi = _get_kiwi(self.user_words)
        out = []
        for toks in kiwi.tokenize([t[: self.max_chars] for t in texts]):
            out.append([(tk.form, tk.tag, tk.sent_position, tk.start, tk.end, tk.lemma) for tk in toks])
        _cache_put(_TOKEN_CACHE, key, out, 6)
        return out

    def terms(self, raw_doc: List[tuple]) -> List[Tuple[str, int]]:
        """[(term, sentence_index)] after compound joining, POS filter, synonyms and stopwords."""
        return [(t, s) for t, s, _ in self.terms_full(raw_doc)]

    def terms_full(self, raw_doc: List[tuple]) -> List[Tuple[str, int, int]]:
        """[(term, sentence_index, character_offset)]; the offset locates paragraphs."""
        allowed = POS_MODES[self.pos_mode]
        stop = self.stopwords
        syn = self.synonyms
        out: List[Tuple[str, int, int]] = []
        i, n = 0, len(raw_doc)
        while i < n:
            form, tag, sent, start, end, lemma = raw_doc[i]
            if self.compound and tag in ("NNG", "NNP"):
                j = i
                parts = [form]
                while j + 1 < n and raw_doc[j + 1][1] in ("NNG", "NNP") and raw_doc[j + 1][3] == raw_doc[j][4] and len(parts) < 4:
                    j += 1
                    parts.append(raw_doc[j][0])
                if len(parts) > 1:
                    form, tag = "".join(parts), "NNG"
                i = j + 1
            else:
                i += 1
            if tag not in allowed:
                continue
            term = (lemma or form + "다") if tag in ("VV", "VA") else form
            term = syn.get(term, term)
            if len(term) < self.min_len or term in stop:
                continue
            out.append((term, sent, start))
        return out


def prep_from_options(o: Dict[str, Any]) -> TextPrep:
    return TextPrep(
        pos_mode=o.get("posMode", "nouns_foreign"), min_len=o.get("minLen", 2), compound=o.get("compound", True),
        default_stop=o.get("defaultStop", True), stopwords=o.get("stopwords", ""), user_words=o.get("userWords", ""),
        synonyms=o.get("synonyms", ""),
    )


PREP_OPTION_KEYS = ("posMode", "minLen", "compound", "defaultStop", "stopwords", "userWords", "synonyms")


class TextCorpus:
    """Valid documents of a text column with their terms."""

    def __init__(self, df: pd.DataFrame, text_col: str, prep: TextPrep, label_col: Optional[str] = None,
                 group_col: Optional[str] = None):
        if text_col not in df.columns:
            raise ValueError(f"텍스트 변수 '{text_col}' 없음. 사용 가능한 변수: {list(df.columns)}")
        s = df[text_col]
        mask = s.notna() & (s.astype(str).str.strip() != "")
        self.n_rows = len(df)
        self.index = list(np.where(mask.values)[0])
        sub = df.iloc[self.index]
        self.texts = [str(t) for t in sub[text_col].tolist()]
        if label_col and label_col in df.columns:
            self.labels = [str(v) for v in sub[label_col].tolist()]
        else:
            self.labels = [re.sub(r"\s+", " ", t).strip()[:30] or f"문서 {i + 1}" for t, i in zip(self.texts, self.index)]
        self.groups = [None if pd.isna(v) else str(v) for v in sub[group_col].tolist()] if group_col and group_col in df.columns else None
        raw = prep.raw_tokens(self.texts)
        full = [prep.terms_full(r) for r in raw]
        self.terms_with_sent = [[(t, s) for t, s, _ in d] for d in full]
        self.term_starts = [[st for _, _, st in d] for d in full]
        self.tokens = [[t for t, _ in d] for d in self.terms_with_sent]
        self.prep = prep

    @property
    def n_docs(self) -> int:
        return len(self.texts)

    def keep(self, keep_mask: Sequence[bool]):
        keep_mask = list(keep_mask)
        sel = [i for i, k in enumerate(keep_mask) if k]
        self.texts = [self.texts[i] for i in sel]
        self.labels = [self.labels[i] for i in sel]
        self.index = [self.index[i] for i in sel]
        if self.groups is not None:
            self.groups = [self.groups[i] for i in sel]
        self.terms_with_sent = [self.terms_with_sent[i] for i in sel]
        self.term_starts = [self.term_starts[i] for i in sel]
        self.tokens = [self.tokens[i] for i in sel]

    def summary_table(self, extra: Optional[List[Tuple[str, float]]] = None) -> ResultsTable:
        tbl = ResultsTable(name="corpus", title="말뭉치 요약")
        tbl.add_column("stat", "지표", col_type="text")
        tbl.add_column("value", "값", col_type="number", fmt="auto")
        total = sum(len(t) for t in self.tokens)
        vocab = len({w for d in self.tokens for w in d})
        rows = [
            ("전체 행 수", self.n_rows), ("분석 문서 수", self.n_docs), ("추출 단어 수 (토큰)", total),
            ("고유 단어 수", vocab), ("문서당 평균 단어 수", total / max(self.n_docs, 1)),
        ] + list(extra or [])
        for k, v in rows:
            tbl.add_row({"stat": k, "value": float(v)})
        tbl.add_note("전처리: " + self.prep.describe() + ".")
        if self.n_docs < self.n_rows:
            tbl.add_note(f"빈 텍스트 {self.n_rows - self.n_docs}행 제외.")
        return tbl


def doc_term_counts(tokens: List[List[str]], min_df: int = 1):
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(analyzer=lambda d: d, min_df=max(int(min_df), 1))
    X = vec.fit_transform(tokens)
    return X, np.array(vec.get_feature_names_out())


def ctfidf_scores(X, labels: Sequence[Any], groups: Sequence[Any], min_total: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Class-based TF-IDF: (count / class total) * log(1 + mean class size / term total)."""
    labels = np.asarray(labels, dtype=object)
    C = np.vstack([np.asarray(X[labels == g].sum(axis=0)).ravel() for g in groups]).astype(float)
    total = C.sum(axis=0)
    avg = C.sum(axis=1).mean() if len(groups) else 0.0
    tf = C / np.clip(C.sum(axis=1, keepdims=True), 1.0, None)
    scores = tf * np.log(1.0 + avg / np.clip(total, 1.0, None))
    scores[:, total < min_total] = 0.0
    return scores, C


def top_terms(scores_row: np.ndarray, vocab: np.ndarray, k: int) -> List[str]:
    order = np.lexsort((vocab, -scores_row))
    return [vocab[i] for i in order[:k] if scores_row[i] > 0]


def korean_font_file(bold: bool = True) -> Optional[str]:
    import matplotlib.font_manager as fm
    prefs = ["KoPubDotum Bold", "KoPubWorldDotum", "KoPubDotum", "NanumSquare", "NanumGothic", "NanumBarunGothic",
             "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "UnDotum"]
    target = 700 if bold else 400
    named = {"thin": 100, "light": 300, "normal": 400, "regular": 400, "medium": 500, "semibold": 600, "bold": 700,
             "heavy": 800, "black": 900}
    for p in prefs:
        cands = [f for f in fm.fontManager.ttflist if f.name == p]
        if cands:
            def _w(f):
                w = f.weight
                return w if isinstance(w, (int, float)) else named.get(str(w).lower(), 400)
            cands.sort(key=lambda f: (abs(_w(f) - target), "thin" in f.fname.lower()))
            return cands[0].fname
    return None


def plot_wordcloud(weights: Dict[str, float], title: str, max_words: int = 120, shape: str = "rect", seed: int = 42,
                   ax=None):
    try:
        from wordcloud import WordCloud
    except ImportError as e:  # pragma: no cover
        raise ValueError("워드클라우드에 wordcloud 패키지 필요: 'pip install wordcloud' 실행 후 다시 시도") from e
    st = get_active_style()
    dark = st.get("theme") == "dark"
    pal = _pal()
    rng = np.random.RandomState(seed)

    def color_func(*_args, **_kw):
        return pal[rng.randint(0, len(pal))]

    mask = None
    w, h = 1500, 900
    if shape == "circle":
        w = h = 1100
        yy, xx = np.ogrid[:h, :w]
        mask = (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) > (w / 2 - 10) ** 2).astype(np.uint8) * 255
    wc = WordCloud(font_path=korean_font_file(True), width=w, height=h, mask=mask, max_words=int(max_words),
                   background_color="#0F172A" if dark else "white", color_func=color_func, random_state=seed,
                   prefer_horizontal=0.92, relative_scaling=0.45, margin=4, collocations=False)
    wc.generate_from_frequencies({k: float(v) for k, v in weights.items() if v > 0})
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8.4, 5.4) if shape != "circle" else (6.8, 6.8))
    ax.imshow(wc.to_array(), interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#F1F5F9" if dark else "#1E293B", pad=8)
    if own:
        fig.tight_layout()
        b64 = fig_to_base64(fig, dpi=160)
        plt.close(fig)
        return b64
    return None


def plot_keyword_bars(terms: List[str], values: List[float], title: str, xlabel: str) -> str:
    st = get_active_style()
    fg = "#F1F5F9" if st.get("theme") == "dark" else "#1E293B"
    n = len(terms)
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * n + 1.4))
    ax.barh(terms[::-1], values[::-1], color=_pal(1)[0], alpha=0.9)
    for y, v in enumerate(values[::-1]):
        ax.text(v, y, f" {v:,.0f}" if float(v).is_integer() else f" {v:.2f}", va="center", fontsize=8, color=fg)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold", color=fg)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def _base_text_options(text, group, doc_label, kw) -> Dict[str, Any]:
    opts = {"text": text, "group": group or "", "docLabel": doc_label or ""}
    for k in PREP_OPTION_KEYS:
        if k in kw:
            opts[k] = kw[k]
    return opts


# ---------------------------------------------------------------------------
# 1. Keyword frequency / TF-IDF / word clouds
# ---------------------------------------------------------------------------
class TextKeywords(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, text: str, group: Optional[str] = None, rank_by: str = "freq",
                 top_n: int = 30, group_top_n: int = 10, wordcloud: bool = True, cloud_weight: str = "freq",
                 max_words: int = 120, cloud_shape: str = "rect", bar_plot: bool = True, group_cloud: bool = False,
                 seed: int = 42, **prep_kw):
        rank_by = resolve_choice("정렬 기준", rank_by, ["freq", "tfidf", "df"],
                                 {"frequency": "freq", "count": "freq", "tf-idf": "tfidf", "tf_idf": "tfidf",
                                  "docfreq": "df", "doc_freq": "df", "document_frequency": "df"}, default="freq")
        cloud_weight = resolve_choice("워드클라우드 가중치", cloud_weight, ["freq", "tfidf"],
                                      {"frequency": "freq", "count": "freq", "tf-idf": "tfidf", "tf_idf": "tfidf"}, default="freq")
        cloud_shape = resolve_choice("워드클라우드 모양", cloud_shape, ["rect", "circle"],
                                     {"rectangle": "rect", "square": "rect", "round": "circle", "ellipse": "circle"}, default="rect")
        opts = _base_text_options(text, group, None, prep_kw)
        opts.update({"rankBy": rank_by, "topN": top_n, "groupTopN": group_top_n, "wordcloud": wordcloud,
                     "cloudWeight": cloud_weight, "maxWords": max_words, "cloudShape": cloud_shape,
                     "barPlot": bar_plot, "groupCloud": group_cloud, "seed": seed})
        super().__init__(name="textKeywords", title="키워드 분석 (빈도·TF-IDF·워드클라우드)", dataset=data, options=opts)
        self.syntax_python = f"sess.text_keywords(text={text!r}" + (f", group={group!r}" if group else "") + ")"
        self.syntax_r = "tidytext::unnest_tokens() |> dplyr::count(word, sort = TRUE)"
        self.run()

    def run(self):
        o = self.options
        prep = prep_from_options(o)
        corpus = TextCorpus(self.dataset, o["text"], prep, group_col=o.get("group") or None)
        if corpus.n_docs == 0:
            raise ValueError("키워드 분석: 분석할 텍스트 없음")
        if sum(len(t) for t in corpus.tokens) == 0:
            raise ValueError("키워드 분석: 전처리 후 남은 단어 없음. 품사·불용어·최소 글자 수 설정 확인 필요")
        X, vocab = doc_term_counts(corpus.tokens)
        from sklearn.feature_extraction.text import TfidfTransformer
        T = TfidfTransformer(sublinear_tf=True).fit_transform(X)
        freq = np.asarray(X.sum(axis=0)).ravel()
        dfreq = np.asarray((X > 0).sum(axis=0)).ravel()
        tfidf = np.asarray(T.sum(axis=0)).ravel()
        key = {"freq": freq, "tfidf": tfidf, "df": dfreq}.get(o["rankBy"], freq)
        order = np.lexsort((vocab, -freq, -key))
        top_n = max(int(o["topN"]), 1)

        self.add_table(corpus.summary_table())

        tbl = ResultsTable(name="keywords", title=f"상위 키워드 {min(top_n, len(vocab))}개")
        tbl.add_column("rank", "순위", col_type="integer")
        tbl.add_column("term", "키워드", col_type="text")
        tbl.add_column("freq", "빈도", col_type="integer")
        tbl.add_column("df", "문서 빈도", col_type="integer")
        tbl.add_column("dfpc", "문서 비율", col_type="number", fmt="pc")
        tbl.add_column("tfidf", "TF-IDF 합계", col_type="number")
        for r, i in enumerate(order[:top_n]):
            tbl.add_row({"rank": r + 1, "term": vocab[i], "freq": int(freq[i]), "df": int(dfreq[i]),
                         "dfpc": 100.0 * dfreq[i] / corpus.n_docs, "tfidf": float(tfidf[i])})
        tbl.add_note({"freq": "정렬: 빈도", "tfidf": "정렬: TF-IDF 합계 (로그 빈도 가중)", "df": "정렬: 문서 빈도"}.get(o["rankBy"], "정렬: 빈도")
                     + ". 문서 비율 = 해당 키워드가 나온 문서 / 전체 문서.")
        self.add_table(tbl)

        groups = []
        if corpus.groups is not None:
            groups = [g for g in pd.unique(pd.Series([g for g in corpus.groups if g is not None]))]
            if len(groups) > 30:
                groups = list(pd.Series(corpus.groups).value_counts().index[:30])
        if len(groups) >= 2:
            labels = np.array([g if g is not None else "__NA__" for g in corpus.groups], dtype=object)
            scores, C = ctfidf_scores(X, labels, groups, min_total=1)
            gtab = ResultsTable(name="groupKeywords", title=f"집단별 특징 키워드 ({o['group']})")
            gtab.add_column("group", "집단", col_type="text")
            gtab.add_column("ndocs", "문서 수", col_type="integer")
            gtab.add_column("rank", "순위", col_type="integer")
            gtab.add_column("term", "키워드", col_type="text")
            gtab.add_column("freq", "집단 내 빈도", col_type="integer")
            gtab.add_column("score", "c-TF-IDF", col_type="number")
            k = max(int(o["groupTopN"]), 1)
            for gi, g in enumerate(groups):
                terms = top_terms(scores[gi], vocab, k)
                idx = {t: j for j, t in enumerate(vocab)}
                nd = int((labels == g).sum())
                for r, t in enumerate(terms):
                    j = idx[t]
                    gtab.add_row({"group": g if r == 0 else "", "ndocs": nd if r == 0 else "", "rank": r + 1, "term": t,
                                  "freq": int(C[gi, j]), "score": float(scores[gi, j])})
            gtab.add_note("c-TF-IDF: 다른 집단보다 해당 집단에서 상대적으로 자주 쓰인 단어일수록 높음.")
            self.add_table(gtab)

        if o["wordcloud"]:
            wsrc = tfidf if o["cloudWeight"] == "tfidf" else freq
            wtop = np.argsort(-wsrc)[: int(o["maxWords"])]
            self.add_plot("워드클라우드", plot_wordcloud({vocab[i]: float(wsrc[i]) for i in wtop},
                                                     title="워드클라우드 (" + ("TF-IDF" if o["cloudWeight"] == "tfidf" else "빈도") + " 기준)",
                                                     max_words=int(o["maxWords"]), shape=o["cloudShape"], seed=int(o["seed"])))
        if o["barPlot"]:
            sel = order[: min(top_n, 25)]
            vals = [float(key[i]) for i in sel]
            self.add_plot("상위 키워드 막대도표", plot_keyword_bars([vocab[i] for i in sel], vals, title="상위 키워드",
                                                             xlabel={"freq": "빈도", "tfidf": "TF-IDF 합계", "df": "문서 빈도"}.get(o["rankBy"], "빈도")))
        if o["groupCloud"] and len(groups) >= 2:
            labels = np.array([g if g is not None else "__NA__" for g in corpus.groups], dtype=object)
            scores, _ = ctfidf_scores(X, labels, groups)
            shown = groups[:6]
            ncol = 3 if len(shown) > 4 else 2
            nrow = int(np.ceil(len(shown) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow))
            axes = np.atleast_1d(axes).ravel()
            for gi, g in enumerate(shown):
                row = scores[gi]
                top = np.argsort(-row)[:60]
                plot_wordcloud({vocab[i]: float(row[i]) for i in top if row[i] > 0}, title=str(g), max_words=60,
                               seed=int(o["seed"]), ax=axes[gi])
            for ax in axes[len(shown):]:
                ax.axis("off")
            fig.suptitle("집단별 워드클라우드 (c-TF-IDF)", fontsize=12.5, fontweight="bold")
            fig.tight_layout()
            self.add_plot("집단별 워드클라우드", fig_to_base64(fig, dpi=150))
            plt.close(fig)

        top3 = ", ".join(f"'{vocab[i]}'({int(freq[i])}회)" for i in order[:3])
        self.apa_narrative = (f"문서 {corpus.n_docs}건에서 고유 단어 {len(vocab)}개를 추출하였다. 상위 키워드는 {top3}이다."
                              + (f" '{o['group']}' 집단별 특징 키워드는 c-TF-IDF로 산출하였다." if len(groups) >= 2 else ""))


# ---------------------------------------------------------------------------
# 2. Keyword co-occurrence network
# ---------------------------------------------------------------------------
UNIT_LABELS = {"sentence": "문장", "paragraph": "단락", "document": "문서"}
EDGE_METRIC_LABELS = {"count": "동시출현 빈도", "jaccard": "Jaccard 계수", "cosine": "코사인 계수", "npmi": "NPMI"}


def paragraph_index(text: str, starts: Sequence[int]) -> List[int]:
    """Paragraph number of each character offset; a paragraph ends at a line break (runs of breaks count once)."""
    cuts = [m.start() for m in re.finditer(r"[ \t]*\r?\n\s*", text)]
    return [bisect.bisect_right(cuts, st) for st in starts]


class TextNetwork(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, text: str, unit: str = "sentence", top_nodes: int = 60, min_cooc: int = 2,
                 edge_metric: str = "count", max_edges: int = 150, community: bool = True, resolution: float = 1.0,
                 layout: str = "spring", label_top: int = 40, top_pairs: int = 30, seed: int = 42, **prep_kw):
        unit = resolve_choice("분석 단위", unit, list(UNIT_LABELS),
                              {"sent": "sentence", "sentences": "sentence", "para": "paragraph", "paragraphs": "paragraph",
                               "doc": "document", "documents": "document"}, default="sentence")
        edge_metric = resolve_choice("연결 강도", edge_metric, list(EDGE_METRIC_LABELS),
                                     {"freq": "count", "frequency": "count", "cooc": "count", "cooccurrence": "count",
                                      "salton": "cosine", "ochiai": "cosine", "cos": "cosine"}, default="count")
        layout = resolve_layout(layout)
        opts = _base_text_options(text, None, None, prep_kw)
        opts.update({"unit": unit, "topNodes": top_nodes, "minCooc": min_cooc, "edgeMetric": edge_metric,
                     "maxEdges": max_edges, "community": community, "resolution": resolution, "layout": layout,
                     "labelTop": label_top, "topPairs": top_pairs, "seed": seed})
        super().__init__(name="textNetwork", title="키워드 동시출현 관계망", dataset=data, options=opts)
        self.syntax_python = f"sess.text_network(text={text!r}, unit={unit!r})"
        self.syntax_r = "widyr::pairwise_count(word, unit) |> igraph::graph_from_data_frame()"
        self.run()

    def run(self):
        nx = require_networkx()
        o = self.options
        corpus = TextCorpus(self.dataset, o["text"], prep_from_options(o))
        if corpus.n_docs == 0:
            raise ValueError("동시출현 관계망: 분석할 텍스트 없음")
        freq = Counter(w for d in corpus.tokens for w in d)
        if len(freq) < 3:
            raise ValueError("동시출현 관계망: 전처리 후 단어가 3개 미만. 전처리 설정 확인 필요")
        keep = [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[: max(int(o["topNodes"]), 3)]]
        keep_set = set(keep)

        units: List[set] = []
        for d, txt, starts in zip(corpus.terms_with_sent, corpus.texts, corpus.term_starts):
            if o["unit"] == "document":
                units.append({t for t, _ in d if t in keep_set})
            elif o["unit"] == "paragraph":
                by_para: Dict[int, set] = {}
                for (t, _), q in zip(d, paragraph_index(txt, starts)):
                    if t in keep_set:
                        by_para.setdefault(q, set()).add(t)
                units.extend(by_para.values())
            else:
                by_sent: Dict[int, set] = {}
                for t, s in d:
                    if t in keep_set:
                        by_sent.setdefault(s, set()).add(t)
                units.extend(by_sent.values())
        n_units = len(units)
        node_df = Counter()
        pair = Counter()
        for u in units:
            s = sorted(u)
            node_df.update(s)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    pair[(s[i], s[j])] += 1

        rows = []
        for (a, b), c in pair.items():
            if c < int(o["minCooc"]):
                continue
            jac = c / max(node_df[a] + node_df[b] - c, 1)
            cos = c / np.sqrt(max(node_df[a] * node_df[b], 1))
            pab, pa, pb = c / n_units, node_df[a] / n_units, node_df[b] / n_units
            npmi = np.log(pab / (pa * pb)) / -np.log(pab) if pab < 1 else 1.0
            rows.append((a, b, c, jac, float(cos), float(npmi)))
        if not rows:
            raise ValueError(f"동시출현 관계망: 최소 동시출현 {o['minCooc']}회 이상인 단어 쌍 없음. 기준을 낮추거나 분석 단위를 '문서'로 변경 필요")
        edges = pd.DataFrame(rows, columns=["a", "b", "count", "jaccard", "cosine", "npmi"])
        metric = o["edgeMetric"]
        if metric == "npmi":
            edges = edges[edges["npmi"] > 0]
        edges = edges.sort_values([metric, "count"], ascending=False).head(max(int(o["maxEdges"]), 1))

        G = nx.Graph()
        for r in edges.itertuples(index=False):
            G.add_edge(r.a, r.b, weight=float(getattr(r, metric)), count=int(r.count))
        df_cent = compute_centrality(G)
        membership = detect_communities(G, resolution=float(o["resolution"]), seed=int(o["seed"])) if o["community"] else {}
        modularity = community_modularity(G, membership) if membership else float("nan")

        unit_label = UNIT_LABELS[o["unit"]]
        summ = corpus.summary_table(extra=[
            (f"분석 단위 수 ({unit_label})", n_units), ("관계망 노드 수", G.number_of_nodes()),
            ("관계망 연결 수", G.number_of_edges()), ("밀도", nx.density(G)),
        ] + ([("묶음(커뮤니티) 수", len(set(membership.values()))), ("모듈성 Q", modularity)] if membership else []))
        summ.title = "말뭉치·관계망 요약"
        summ.add_note(f"동시출현: 같은 {unit_label} 안에 함께 나온 횟수. 상위 {len(keep)}개 단어 대상, "
                      f"최소 {o['minCooc']}회, 연결 강도 = {EDGE_METRIC_LABELS[metric]}, "
                      f"상위 {len(edges)}개 연결 표시.")
        if o["unit"] == "paragraph" and n_units <= corpus.n_docs:
            summ.add_note("문서에 줄바꿈이 거의 없어 단락 단위가 문서 단위와 사실상 같음.")
        if metric == "cosine":
            summ.add_note("코사인 계수 = 동시출현 / √(단어 A 출현 단위 수 × 단어 B 출현 단위 수) (Salton 코사인).")
        self.add_table(summ)

        add_centrality_table(self, df_cent, membership, sort_by="strength", title="키워드 중심성", node_title="키워드",
                             extra_cols={"freq": {"title": "빈도", "type": "integer", "values": dict(freq)}})

        ptab = ResultsTable(name="pairs", title="주요 동시출현 단어 쌍")
        ptab.add_column("rank", "순위", col_type="integer")
        ptab.add_column("a", "단어 1", col_type="text")
        ptab.add_column("b", "단어 2", col_type="text")
        ptab.add_column("count", "동시출현 빈도", col_type="integer")
        ptab.add_column("jaccard", "Jaccard", col_type="number")
        ptab.add_column("cosine", "코사인", col_type="number")
        ptab.add_column("npmi", "NPMI", col_type="number")
        for r, row in enumerate(edges.head(max(int(o["topPairs"]), 1)).itertuples(index=False)):
            ptab.add_row({"rank": r + 1, "a": row.a, "b": row.b, "count": int(row.count), "jaccard": float(row.jaccard),
                          "cosine": float(row.cosine), "npmi": float(row.npmi)})
        self.add_table(ptab)

        names = {}
        if membership:
            strength = dict(zip(df_cent["node"], df_cent["strength"]))
            ctab = ResultsTable(name="communities", title="키워드 묶음(커뮤니티)")
            ctab.add_column("community", "묶음", col_type="integer")
            ctab.add_column("n", "단어 수", col_type="integer")
            ctab.add_column("members", "주요 단어 (가중 연결정도 순)", col_type="text")
            for c in sorted(set(membership.values())):
                mem = sorted([n for n, cc in membership.items() if cc == c], key=lambda n: -strength.get(n, 0))
                names[c] = f"묶음 {c}: " + "·".join(mem[:3])
                ctab.add_row({"community": c, "n": len(mem), "members": ", ".join(mem[:15]) + (" …" if len(mem) > 15 else "")})
            self.add_table(ctab)

        self.add_plot("키워드 동시출현 관계망", plot_network_map(
            G, title=f"키워드 동시출현 관계망 ({unit_label} 단위)", size_values={n: freq[n] for n in G.nodes()},
            membership=membership, community_names=names, label_top=int(o["labelTop"]), layout=o["layout"],
            seed=int(o["seed"])))

        top = df_cent.sort_values("strength", ascending=False).head(3)["node"].tolist()
        self.apa_narrative = (f"{unit_label} 단위 동시출현 관계망은 단어 {G.number_of_nodes()}개와 연결 {G.number_of_edges()}개로 구성된다. "
                              f"가중 연결정도 상위 단어는 {', '.join(repr(t) for t in top)}이다."
                              + (f" Louvain 묶음 {len(set(membership.values()))}개 (모듈성 Q = {modularity:.3f})." if membership else ""))


# ---------------------------------------------------------------------------
# 3. Document embedding + clustering
# ---------------------------------------------------------------------------
def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def tfidf_lsa_embedding(tokens: List[List[str]], min_df: int, seed: int) -> Tuple[np.ndarray, int]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    n = len(tokens)
    vec = TfidfVectorizer(analyzer=lambda d: d, min_df=max(int(min_df), 1) if n >= 10 else 1,
                          max_df=0.95 if n >= 20 else 1.0, sublinear_tf=True)
    T = vec.fit_transform(tokens)
    V = T.shape[1]
    k = min(100, V - 1, n - 1)
    if k >= 2:
        X = TruncatedSVD(n_components=k, random_state=seed).fit_transform(T)
    else:
        X = T.toarray()
    return _l2(X), V


def sbert_embedding(texts: List[str], model_name: str, max_len: int = 512) -> np.ndarray:
    h = hashlib.sha1((model_name + f"|{max_len}").encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8", "ignore"))
    key = h.hexdigest()
    if key in _EMBED_CACHE:
        _EMBED_CACHE.move_to_end(key)
        return _EMBED_CACHE[key]
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ValueError("임베딩 모델 사용에 sentence-transformers·torch 필요: 'pip install -r requirements-embedding.txt' 실행 후 다시 시도. "
                         "기본 방식(TF-IDF + LSA)은 추가 설치 없이 사용 가능") from e
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise ValueError(f"임베딩 모델 '{model_name}' 불러오기 실패: {e}") from e
        _MODEL_CACHE.clear()
        _MODEL_CACHE[model_name] = model
    model.max_seq_length = max_len
    X = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    X = _l2(np.asarray(X))
    _cache_put(_EMBED_CACHE, key, X, 4)
    return X


def _reduce(X: np.ndarray, use_umap: bool, n_neighbors: int, n_components: int, min_dist: float, seed: int) -> np.ndarray:
    n = X.shape[0]
    if use_umap and n >= 15:
        import warnings
        import umap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
            return umap.UMAP(n_neighbors=max(2, min(int(n_neighbors), n - 1)), n_components=max(2, min(n_components, n - 2)),
                             min_dist=min_dist, metric="cosine", random_state=seed, n_jobs=1).fit_transform(X)
    if n_components == 2:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    return X


def _auto_k(Z: np.ndarray, method: str, seed: int, k_max: int = 10) -> Tuple[int, float]:
    from sklearn.metrics import silhouette_score
    n = Z.shape[0]
    best = (2, -1.0)
    for k in range(2, max(3, min(k_max, n - 1) + 1)):
        lab = _fit_flat(Z, method, k, seed)
        if len(set(lab)) < 2:
            continue
        s = float(silhouette_score(Z, lab))
        if s > best[1] + 1e-9:
            best = (k, s)
    return best


def _fit_flat(Z: np.ndarray, method: str, k: int, seed: int) -> np.ndarray:
    if method in ("agglomerative", "hierarchical"):
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def _cluster_once(X: np.ndarray, o: Dict[str, Any], seed: int, notes: List[str]) -> Tuple[np.ndarray, np.ndarray, str]:
    """Returns (labels with -1 = noise, reduced matrix, method actually used)."""
    n = X.shape[0]
    Z = _reduce(X, bool(o["useUmap"]), int(o["nNeighbors"]), 5, 0.0, seed)
    method = "agglomerative" if o["method"] == "hierarchical" else o["method"]
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        mcs = int(o.get("minClusterSize") or 0)
        if mcs <= 0:
            mcs = max(3, min(10, n // 30))
        ms = max(1, min(4, mcs))
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method="eom", copy=True).fit_predict(Z)
        if len(set(lab) - {-1}) >= 2:
            if seed == int(o["seed"]):
                notes.append(f"HDBSCAN: 최소 군집 크기 {mcs}, min_samples {ms}, eom.")
            return lab, Z, "hdbscan"
        if seed == int(o["seed"]):
            notes.append("HDBSCAN에서 군집이 2개 미만으로 나와 k-평균(실루엣 기준 자동 군집 수)으로 대체.")
        method = "kmeans"
    k = int(o.get("nClusters") or 0)
    if k <= 0 or k >= n:
        k, sil = _auto_k(Z, method, seed)
        if seed == int(o["seed"]):
            notes.append(f"군집 수 자동 선택: k = {k} (실루엣 {sil:.3f}, 후보 2~{min(10, n - 1)}).")
    return _fit_flat(Z, method, k, seed), Z, method


class TextCluster(BaseAnalysis):
    def __init__(self, data: pd.DataFrame, text: str, doc_label: Optional[str] = None, group: Optional[str] = None,
                 embedding: str = "tfidf", model: str = "BAAI/bge-m3", method: str = "hdbscan", n_clusters: int = 0,
                 min_cluster_size: int = 0, use_umap: bool = True, n_neighbors: int = 12, assign_noise: bool = True,
                 noise_assign_min: float = 0.6, themes: bool = True, theme_distance: float = 0.3, stability: bool = False,
                 kw_top: int = 8, min_df: int = 2, cluster_map: bool = True, cluster_network: bool = True,
                 doc_network: bool = False, doc_sim_min: float = 0.6, doc_table: bool = True, seed: int = 42, **prep_kw):
        embedding = resolve_choice("문서 표현", embedding, ["tfidf", "sbert"],
                                   {"lsa": "tfidf", "tf-idf": "tfidf", "tf_idf": "tfidf", "bge": "sbert", "bge-m3": "sbert",
                                    "embedding": "sbert", "sentence": "sbert", "sentence_embedding": "sbert"}, default="tfidf")
        method = resolve_choice("군집 방법", method, ["hdbscan", "kmeans", "agglomerative"],
                                {"hierarchical": "agglomerative", "ward": "agglomerative", "k-means": "kmeans", "k_means": "kmeans"},
                                default="hdbscan")
        opts = _base_text_options(text, group, doc_label, prep_kw)
        opts.update({"embedding": embedding, "model": model, "method": method, "nClusters": n_clusters,
                     "minClusterSize": min_cluster_size, "useUmap": use_umap, "nNeighbors": n_neighbors,
                     "assignNoise": assign_noise, "noiseAssignMin": noise_assign_min, "themes": themes,
                     "themeDistance": theme_distance, "stability": stability, "kwTop": kw_top, "minDf": min_df,
                     "clusterMap": cluster_map, "clusterNetwork": cluster_network, "docNetwork": doc_network,
                     "docSimMin": doc_sim_min, "docTable": doc_table, "seed": seed})
        super().__init__(name="textCluster", title="문서 군집 분석 (임베딩·주제 분류)", dataset=data, options=opts)
        self.syntax_python = f"sess.text_cluster(text={text!r}, embedding={embedding!r}, method={method!r})"
        self.syntax_r = "BERTopic-style pipeline: embed -> umap -> hdbscan -> c-TF-IDF"
        self.run()

    def run(self):
        o = self.options
        seed = int(o["seed"])
        notes: List[str] = []
        corpus = TextCorpus(self.dataset, o["text"], prep_from_options(o), label_col=o.get("docLabel") or None,
                            group_col=o.get("group") or None)
        if o["embedding"] == "tfidf":
            corpus.keep([len(t) > 0 for t in corpus.tokens])
        if corpus.n_docs < 6:
            raise ValueError(f"문서 군집 분석: 유효 문서가 {corpus.n_docs}건. 최소 6건 필요")

        if o["embedding"] == "sbert":
            model_name = str(o.get("model") or "BAAI/bge-m3").strip()
            X = sbert_embedding(corpus.texts, model_name)
            emb_desc = f"문장 임베딩 모델 {model_name} (정규화 벡터 {X.shape[1]}차원)"
        else:
            X, V = tfidf_lsa_embedding(corpus.tokens, int(o["minDf"]), seed)
            emb_desc = f"TF-IDF + LSA (어휘 {V}개 → {X.shape[1]}차원)"
        n = X.shape[0]

        labels, Z, used = _cluster_once(X, o, seed, notes)
        base_labels = labels.copy()

        clus = [c for c in pd.Series(labels[labels >= 0]).value_counts().index]
        remap = {c: i + 1 for i, c in enumerate(clus)}
        final = np.array([remap.get(c, 0) for c in labels], dtype=int)
        K = len(clus)
        cents = _l2(np.vstack([X[final == c].mean(axis=0) for c in range(1, K + 1)]))

        n_noise_before = int((final == 0).sum())
        if used == "hdbscan" and n_noise_before and o["assignNoise"]:
            nz = np.where(final == 0)[0]
            sims = X[nz] @ cents.T
            j = sims.argmax(axis=1)
            ok = sims[np.arange(len(nz)), j] >= float(o["noiseAssignMin"])
            final[nz[ok]] = j[ok] + 1
            notes.append(f"미분류 {n_noise_before}건 중 {int(ok.sum())}건을 코사인 유사도 {o['noiseAssignMin']} 이상인 가장 가까운 군집에 재배정.")
            cents = _l2(np.vstack([X[final == c].mean(axis=0) for c in range(1, K + 1)]))
        n_noise = int((final == 0).sum())

        stab = None
        if o["stability"]:
            from sklearn.metrics import adjusted_rand_score
            aris = []
            for s in STABILITY_SEEDS:
                if s == seed:
                    continue
                lab_s, _, _ = _cluster_once(X, o, s, [])
                aris.append(float(adjusted_rand_score(base_labels, lab_s)))
            stab = (float(np.mean(aris)), aris)

        sil = float("nan")
        mask = final > 0
        if K >= 2 and mask.sum() > K:
            from sklearn.metrics import silhouette_score
            try:
                sil = float(silhouette_score(Z[mask], final[mask]))
            except Exception:
                pass

        Xc, vocab = doc_term_counts(corpus.tokens)
        lab_str = np.array([str(c) for c in final], dtype=object)
        groups = [str(c) for c in range(1, K + 1)]
        min_total = max(int(o["minDf"]), 1) if n >= 30 else 1
        scores, _ = ctfidf_scores(Xc, lab_str, groups, min_total=min_total)
        kw_top = max(int(o["kwTop"]), 1)
        kws = {c: top_terms(scores[c - 1], vocab, kw_top) for c in range(1, K + 1)}

        theme_of: Dict[int, int] = {}
        theme_kws: Dict[int, List[str]] = {}
        if o["themes"] and K >= 3:
            from sklearn.cluster import AgglomerativeClustering
            th = AgglomerativeClustering(n_clusters=None, distance_threshold=float(o["themeDistance"]), metric="cosine",
                                         linkage="average").fit_predict(cents)
            order_t = [t for t in pd.Series(th).value_counts().index]
            tmap = {t: i + 1 for i, t in enumerate(order_t)}
            theme_of = {c + 1: tmap[t] for c, t in enumerate(th)}
            tlab = np.array([str(theme_of.get(int(c), 0)) for c in final], dtype=object)
            tgroups = [str(t) for t in range(1, len(order_t) + 1)]
            tsc, _ = ctfidf_scores(Xc, tlab, tgroups, min_total=min_total)
            theme_kws = {int(t): top_terms(tsc[i], vocab, kw_top) for i, t in enumerate(tgroups)}
            if len(order_t) == K:
                notes.append(f"주제 묶음: 코사인 거리 {o['themeDistance']} 이내로 결합되는 군집 없음 (군집 = 주제).")
                theme_of, theme_kws = {}, {}
            elif len(order_t) == 1:
                notes.append(f"주제 묶음: 모든 군집이 코사인 거리 {o['themeDistance']} 이내로 하나의 주제에 결합됨. "
                             "세분화하려면 '주제 결합 코사인 거리'를 낮출 필요 (문장 임베딩은 0.1~0.2 권장).")
                theme_of, theme_kws = {}, {}

        reps = {}
        for c in range(1, K + 1):
            idx = np.where(final == c)[0]
            sims = X[idx] @ cents[c - 1]
            reps[c] = [corpus.labels[i] for i in idx[np.argsort(-sims)[:3]]]

        # --- Tables ---
        qual = corpus.summary_table(extra=[("군집 수", K), ("미분류 문서 수", n_noise), ("실루엣 계수", sil)]
                                    + ([("안정성 평균 ARI", stab[0])] if stab else []))
        qual.title = "말뭉치·군집 요약"
        qual.add_note("문서 표현: " + emb_desc + ".")
        if o["useUmap"] and n >= 15:
            qual.add_note("실루엣 계수는 UMAP 축소 공간에서 계산 — 원래 고차원 공간보다 높게 나오는 경향이 있어 군집 간 비교용으로 사용.")
        qual.add_note(("UMAP 5차원 축소 (이웃 " + str(o["nNeighbors"]) + ", min_dist 0, 코사인) 후 " if (o["useUmap"] and n >= 15) else "차원 축소 없이 ")
                      + {"hdbscan": "HDBSCAN", "kmeans": "k-평균", "agglomerative": "Ward 계층 군집", "hierarchical": "Ward 계층 군집"}.get(used, used) + " 군집화.")
        for nt in notes:
            qual.add_note(nt)
        if stab:
            qual.add_note("안정성: 시드 " + ", ".join(str(s) for s in STABILITY_SEEDS if s != seed)
                          + f"로 재실행한 결과와의 조정 랜드 지수(ARI) " + ", ".join(f"{a:.3f}" for a in stab[1]) + ".")
        self.add_table(qual)

        ctab = ResultsTable(name="clusters", title="군집별 대표 키워드")
        ctab.add_column("cluster", "군집", col_type="integer")
        if theme_of:
            ctab.add_column("theme", "주제 묶음", col_type="integer")
        ctab.add_column("n", "문서 수", col_type="integer")
        ctab.add_column("pc", "비율", col_type="number", fmt="pc")
        ctab.add_column("keywords", "대표 키워드 (c-TF-IDF)", col_type="text")
        ctab.add_column("reps", "대표 문서 (중심과 가장 가까운 순)", col_type="text")
        for c in range(1, K + 1):
            row = {"cluster": c, "n": int((final == c).sum()), "pc": 100.0 * (final == c).sum() / n,
                   "keywords": ", ".join(kws[c]), "reps": " | ".join(r[:40] for r in reps[c])}
            if theme_of:
                row["theme"] = theme_of[c]
            ctab.add_row(row)
        if n_noise:
            ctab.add_row({"cluster": 0, "n": n_noise, "pc": 100.0 * n_noise / n, "keywords": "(미분류)", "reps": ""})
        ctab.add_note("대표 키워드: 다른 군집보다 해당 군집에서 상대적으로 자주 쓰인 단어 (c-TF-IDF). 군집 0 = 미분류.")
        self.add_table(ctab)

        if theme_of:
            ttab = ResultsTable(name="themes", title="상위 주제 묶음")
            ttab.add_column("theme", "주제 묶음", col_type="integer")
            ttab.add_column("clusters", "포함 군집", col_type="text")
            ttab.add_column("n", "문서 수", col_type="integer")
            ttab.add_column("keywords", "주제 키워드", col_type="text")
            for t in sorted(set(theme_of.values())):
                cs = [c for c, tt in theme_of.items() if tt == t]
                ttab.add_row({"theme": t, "clusters": ", ".join(map(str, cs)), "n": int(np.isin(final, cs).sum()),
                              "keywords": ", ".join(theme_kws.get(t, []))})
            ttab.add_note(f"군집 중심 벡터의 평균 연결 계층 군집 (코사인 거리 {o['themeDistance']} 이하 결합).")
            self.add_table(ttab)

        if corpus.groups is not None:
            gser = pd.Series(corpus.groups, name="group").fillna("(결측)")
            ct = pd.crosstab(gser, pd.Series(final, name="cluster"))
            ct = ct[[c for c in ct.columns]]
            xt = ResultsTable(name="crosstab", title=f"집단 × 군집 교차표 ({o['group']})")
            xt.add_column("group", "집단", col_type="text")
            for c in ct.columns:
                xt.add_column(f"c{c}", f"군집 {c}" if c else "미분류", col_type="integer")
            xt.add_column("total", "합계", col_type="integer")
            for g, r in ct.iterrows():
                row = {"group": str(g), "total": int(r.sum())}
                for c in ct.columns:
                    row[f"c{c}"] = int(r[c])
                xt.add_row(row)
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                from scipy.stats import chi2_contingency
                chi2, p, dof, _ = chi2_contingency(ct.values)
                cv = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
                xt.add_note(f"χ²({dof}) = {chi2:.2f}, p {'< .001' if p < 0.001 else '= ' + format(p, '.3f')}, Cramér's V = {cv:.3f}.")
            self.add_table(xt)

        if o["docTable"]:
            dtab = ResultsTable(name="documents", title="문서별 군집 배정")
            dtab.add_column("row", "행 번호", col_type="integer")
            dtab.add_column("doc", "문서", col_type="text")
            dtab.add_column("cluster", "군집", col_type="integer")
            if theme_of:
                dtab.add_column("theme", "주제 묶음", col_type="integer")
            dtab.add_column("sim", "군집 중심 유사도", col_type="number")
            limit = 2000
            for i in range(min(n, limit)):
                c = int(final[i])
                row = {"row": corpus.index[i] + 1, "doc": corpus.labels[i][:60], "cluster": c,
                       "sim": float(X[i] @ cents[c - 1]) if c > 0 else float("nan")}
                if theme_of:
                    row["theme"] = theme_of.get(c, 0)
                dtab.add_row(row)
            if n > limit:
                dtab.add_note(f"처음 {limit}건만 표시 (전체 {n}건).")
            self.add_table(dtab)

        # --- Plots ---
        self._final, self._labels_text = final, corpus.labels
        if o["clusterMap"]:
            xy = _reduce(X, bool(o["useUmap"]), int(o["nNeighbors"]), 2, 0.1, seed)
            self.add_plot("문서 군집 분포도", self._plot_cluster_map(xy, final, kws, K, used_umap=bool(o["useUmap"]) and n >= 15))
        if o["clusterNetwork"] and K >= 2:
            self.add_plot("군집 관계망", self._plot_cluster_network(cents, final, kws, theme_of, theme_kws, K))
        if o["docNetwork"]:
            if n <= 600:
                self.add_plot("문서 유사도 관계망", self._plot_doc_network(X, final, corpus.labels, float(o["docSimMin"])))
            else:
                qual.add_note("문서 유사도 관계망은 600건 이하에서만 그림.")

        biggest = max(range(1, K + 1), key=lambda c: (final == c).sum()) if K else 0
        self.apa_narrative = (
            f"문서 {n}건을 {emb_desc}로 표현하고 {({'hdbscan': 'HDBSCAN', 'kmeans': 'k-평균', 'agglomerative': 'Ward 계층 군집'})[used]}으로 군집화한 결과 "
            f"군집 {K}개가 도출되었다" + (f"(미분류 {n_noise}건)" if n_noise else "") + ". "
            + (f"가장 큰 군집 {biggest}({int((final == biggest).sum())}건)의 대표 키워드는 {', '.join(kws[biggest][:4])}이다. " if K else "")
            + (f"실루엣 계수는 {sil:.3f}" + (f", 시드 변경 안정성 평균 ARI는 {stab[0]:.3f}" if stab else "") + "이다." if np.isfinite(sil) else "")
        )

    def _plot_cluster_map(self, xy: np.ndarray, final: np.ndarray, kws: Dict[int, List[str]], K: int, used_umap: bool) -> str:
        st = get_active_style()
        dark = st.get("theme") == "dark"
        fg = "#F1F5F9" if dark else "#1E293B"
        halo = "#0F172A" if dark else "#FFFFFF"
        pal = _pal(max(K, 1))
        fig, ax = plt.subplots(figsize=(9.2, 6.6))
        noise = final == 0
        if noise.any():
            ax.scatter(xy[noise, 0], xy[noise, 1], s=10, c="#CBD5E1", alpha=0.6, label="미분류", linewidths=0)
        for c in range(1, K + 1):
            m = final == c
            ax.scatter(xy[m, 0], xy[m, 1], s=18, color=pal[(c - 1) % len(pal)], alpha=0.78, linewidths=0,
                       label=f"{c}. " + "·".join(kws[c][:3]) + f" ({int(m.sum())})")
        for c in range(1, K + 1):
            m = final == c
            cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
            t = ax.text(cx, cy, f"{c}. " + "·".join(kws[c][:2]), fontsize=9.5, fontweight="bold", color=fg, ha="center", va="center", zorder=6)
            t.set_path_effects([path_effects.withStroke(linewidth=3.2, foreground=halo)])
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.2, frameon=False, markerscale=1.6, title="군집 (문서 수)")
        if leg is not None:
            for tx in leg.get_texts():
                tx.set_color(fg)
            leg.get_title().set_color(fg)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title("문서 군집 분포도 (" + ("UMAP" if used_umap else "PCA") + " 2차원)", fontsize=12.5, fontweight="bold", color=fg)
        fig.tight_layout()
        b64 = fig_to_base64(fig)
        plt.close(fig)
        return b64

    def _plot_cluster_network(self, cents, final, kws, theme_of, theme_kws, K) -> str:
        nx = require_networkx()
        S = cents @ cents.T
        G = nx.Graph()
        for c in range(1, K + 1):
            G.add_node(c)
        for i in range(K):
            order = [j for j in np.argsort(-S[i]) if j != i][:2]
            for j in order:
                G.add_edge(i + 1, int(j) + 1, weight=float(max(S[i, j], 0.01)))
        sizes = {c: float((final == c).sum()) for c in range(1, K + 1)}
        labels = {c: f"{c}. " + "·".join(kws[c][:2]) for c in range(1, K + 1)}
        names = {t: f"주제 {t}: " + "·".join(theme_kws.get(t, [])[:3]) for t in set(theme_of.values())} if theme_of else None
        return plot_network_map(G, title="군집 관계망 (선 = 군집 중심 유사도, 군집마다 가장 가까운 2개 연결)",
                                size_values=sizes, membership=theme_of or {c: c for c in range(1, K + 1)},
                                community_names=names or {c: labels[c] for c in range(1, K + 1)}, labels=labels,
                                label_top=K, layout="spring", seed=int(self.options["seed"]), size_range=(300.0, 2600.0),
                                legend_title="주제 묶음" if theme_of else "군집")

    def _plot_doc_network(self, X, final, doc_labels, sim_min) -> str:
        nx = require_networkx()
        S = X @ X.T
        np.fill_diagonal(S, -1)
        G = nx.Graph()
        n = X.shape[0]
        for i in range(n):
            G.add_node(i)
            for j in np.argsort(-S[i])[:3]:
                if S[i, j] >= sim_min:
                    G.add_edge(i, int(j), weight=float(S[i, j]))
        G.remove_nodes_from([v for v in list(G.nodes()) if G.degree(v) == 0])
        membership = {i: int(final[i]) for i in G.nodes()}
        labels = {i: doc_labels[i][:14] for i in G.nodes()}
        return plot_network_map(G, title=f"문서 유사도 관계망 (코사인 ≥ {sim_min}, 문서마다 최대 3개 연결)",
                                size_values=dict(G.degree()), membership=membership,
                                community_names={c: (f"군집 {c}" if c else "미분류") for c in set(membership.values())},
                                labels=labels, label_top=12, layout="spring", seed=int(self.options["seed"]), size_range=(30.0, 260.0),
                                legend_title="군집")
