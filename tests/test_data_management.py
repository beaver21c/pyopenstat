"""Data management following Stata semantics: merge / joinby / append / reshape / collapse / egen,
variable type conversion (destring, tostring, encode, decode, value labels), .dta/.sav I/O and Stata transpilation."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "reconstructed"))

import pyjamovi as ost  # noqa: E402
from pyjamovi.data_ops import (  # noqa: E402
    append_datasets, collapse_dataset, convert_variables, decode_variable, egen_variable, encode_variable,
    joinby_datasets, merge_datasets, reshape_dataset,
)
from pyjamovi.stata_transpiler import transpile_stata_line, transpile_stata_script  # noqa: E402

L = pd.DataFrame({"id": [1, 1, 2, 3], "x": [10, 11, 20, 30], "z": [np.nan, 1.0, 2.0, 3.0]})
R_DUP = pd.DataFrame({"id": [1, 2, 2, 4], "z": [9.0, 8.0, 7.0, 6.0], "w": ["a", "b", "b2", "d"]})
R = pd.DataFrame({"id": [1, 2, 4], "z": [9.0, 8.0, 6.0], "w": ["a", "b", "d"]})


def test_merge_cardinality_is_checked():
    with pytest.raises(ValueError, match="r\\(459\\)"):
        merge_datasets(L, R_DUP, on="id")                      # both sides duplicated: no silent expansion
    with pytest.raises(ValueError, match="1:1"):
        merge_datasets(L, R, on="id", kind="1:1")
    with pytest.raises(ValueError, match="m:1"):
        merge_datasets(L, R_DUP, on="id", kind="m:1")
    with pytest.raises(ValueError, match="키\\(on\\)"):
        merge_datasets(L, R)                                   # key is required (no first-common-column guess)
    with pytest.raises(ValueError):
        merge_datasets(L, R_DUP, on="id", how="left")          # legacy mode gets the same protection
    out, rep = merge_datasets(L, R, on="id", return_report=True)
    assert rep["kind"] == "m:1" and rep["inferred"]
    assert rep["counts"] == {1: 1, 2: 1, 3: 3, 4: 0, 5: 0}
    assert out["_merge"].tolist() == [3, 3, 3, 1, 2]
    assert out["z"].tolist()[:3] == [pytest.approx(np.nan, nan_ok=True), 1.0, 2.0]  # master values kept


def test_merge_update_replace_keep_keepusing_assert():
    up = merge_datasets(L, R, on="id", update=True)
    assert up["_merge"].tolist() == [4, 5, 5, 1, 2]           # 4 missing updated, 5 nonmissing conflict
    assert up["z"].tolist()[:3] == [9.0, 1.0, 2.0]
    rp = merge_datasets(L, R, on="id", update=True, replace=True, keep="match")
    assert rp["z"].tolist() == [9.0, 9.0, 8.0] and len(rp) == 3
    ku = merge_datasets(L, R, on="id", keepusing=["w"], indicator=False)
    assert "_merge" not in ku.columns and list(ku.columns) == ["id", "x", "z", "w"]
    with pytest.raises(ValueError, match="assert"):
        merge_datasets(L, R, on="id", assert_results="match")
    legacy = merge_datasets(L, R, on="id", how="left")
    assert len(legacy) == 4 and "z_merged" in legacy.columns and "_merge" not in legacy.columns
    with pytest.raises(ValueError, match="already defined"):
        merge_datasets(L.assign(_merge=1), R, on="id")


def test_merge_mm_is_sequential_and_joinby_is_cartesian():
    mm = merge_datasets(L, R_DUP, on="id", kind="m:m")
    # id 1: two master rows paired with the single using row (repeated); id 2: master row repeated for 2 using rows
    assert mm.loc[mm.id == 1, "w"].tolist() == ["a", "a"]
    assert mm.loc[mm.id == 2, ["x", "w"]].values.tolist() == [[20, "b"], [20, "b2"]]
    assert sorted(mm["_merge"].tolist()) == [1, 2, 3, 3, 3, 3]
    a = pd.DataFrame({"k": [1, 1], "a": [1, 2]})
    b = pd.DataFrame({"k": [1, 1, 1], "b": [7, 8, 9]})
    assert len(merge_datasets(a, b, on="k", kind="m:m")) == 3
    assert len(joinby_datasets(a, b, on="k")) == 6
    jb = joinby_datasets(L, R_DUP, on="id", unmatched="both")
    assert set(jb["_merge"]) == {1, 2, 3} and len(jb) == 6


def test_append_types_and_generate():
    with pytest.raises(ValueError, match="r\\(106\\)"):
        append_datasets(L, pd.DataFrame({"id": ["a"], "x": [1]}))
    out = append_datasets(L, pd.DataFrame({"id": ["a"], "x": [1]}), force=True)
    assert out["id"].isna().tolist()[-1]
    out = append_datasets(L, [R, R], generate="src")
    assert out["src"].tolist() == [0] * 4 + [1] * 3 + [2] * 3 and list(out.columns)[:3] == ["id", "x", "z"]


def test_reshape_wide_follows_stata_rules_and_round_trips():
    lg = pd.DataFrame({"id": [1, 1, 2, 2], "t": [1, 2, 1, 2], "y": [5, 6, 8, 9], "inc": [100, 120, 200, 210], "sex": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="not constant within i"):
        reshape_dataset(lg, "wide", id_vars=["id"], value_vars=["y"], time_col="t")
    with pytest.raises(ValueError, match="not unique within i"):
        reshape_dataset(pd.concat([lg, lg.head(1)]), "wide", id_vars=["id"], value_vars=["y", "inc"], time_col="t")
    with pytest.raises(ValueError, match="결측"):
        reshape_dataset(lg.assign(id=[1, 1, np.nan, np.nan]), "wide", id_vars=["id"], value_vars=["y", "inc"], time_col="t")
    with pytest.raises(ValueError, match="i\\)"):
        reshape_dataset(lg, "wide", value_vars=["y", "inc"], time_col="t")
    w = reshape_dataset(lg, "wide", id_vars=["id"], value_vars=["y", "inc"], time_col="t")
    assert list(w.columns) == ["id", "y1", "y2", "inc1", "inc2", "sex"]          # Stata naming stub + j
    back = reshape_dataset(w, "long", id_vars=["id"], stubnames=["y", "inc"], time_col="t")
    assert back.sort_values(["id", "t"])[["id", "t", "y", "inc", "sex"]].values.tolist() == lg[["id", "t", "y", "inc", "sex"]].values.tolist()
    with pytest.raises(ValueError, match="uniquely identify"):
        reshape_dataset(pd.DataFrame({"id": [1, 1], "y1": [1, 2], "y2": [3, 4]}), "long", id_vars=["id"], stubnames=["y"], time_col="t")


def test_collapse_and_egen():
    lg = pd.DataFrame({"id": [1, 1, 2, 2], "y": [5, 6, 8, 9], "g": ["a", "a", "b", "c"]})
    c = collapse_dataset(lg, {"mean": ["y"], "ymax": ("y", "max"), "n": ("y", "count")}, by="id")
    assert c.values.tolist() == [[1, 5.5, 6, 2], [2, 8.5, 9, 2]]
    assert egen_variable(lg, "m", "mean", "y", by="id")["m"].tolist() == [5.5, 5.5, 8.5, 8.5]
    assert egen_variable(lg, "k", "group", ["g"])["k"].tolist() == [1, 1, 2, 3]
    assert egen_variable(lg, "tg", "tag", by="id")["tg"].tolist() == [1, 0, 1, 0]
    with pytest.raises(ValueError):
        egen_variable(lg, "y", "mean", "y")


def test_type_conversion_encode_decode():
    t = pd.DataFrame({"a": ["1,234", " 5 ", ".", "x"], "b": ["10%", "5%", None, "1"], "c": [1.0, 2.0, np.nan, 2.5], "s": ["남", "여", "여", None]})
    with pytest.raises(ValueError, match="nonnumeric"):
        convert_variables(t, ["a"], "numeric", ignore=",")
    o, rep = convert_variables(t, ["a"], "numeric", ignore=",", force=True)
    assert o["a"].tolist()[:2] == [1234.0, 5.0] and rep["forced_missing"]["a"] == 1
    o, _ = convert_variables(t, ["b"], "numeric", percent=True)
    assert o["b"].tolist()[:2] == [0.1, 0.05]
    o, _ = convert_variables(t, ["c"], "text")
    assert o["c"].tolist() == ["1", "2", None, "2.5"]
    with pytest.raises(ValueError):
        convert_variables(t, ["c"], "integer")
    o, _ = convert_variables(t, ["c"], "integer", force=True, generate="ci")
    assert o["ci"].tolist()[3] == 3.0                                            # half away from zero
    o, lab = encode_variable(t, "s", "sc")
    assert lab == {1: "남", 2: "여"} and o["sc"].tolist()[:3] == [1.0, 2.0, 2.0]
    assert decode_variable(o, "sc", "st", lab)["st"].tolist() == ["남", "여", "여", None]
    with pytest.raises(ValueError, match="not labeled"):
        decode_variable(o, "c", "ct", None)


def test_session_records_replays_and_writes_labels(tmp_path):
    m = pd.DataFrame({"id": [1, 2, 3, 4], "성별": ["남", "여", "여", "남"], "소득": ["1,200", "3,400", "n/a", "2,100"], "z": [1.0, np.nan, 3.0, 4.0]})
    u = pd.DataFrame({"id": [1, 2, 5], "지역": ["서울", "부산", "대구"], "z": [9.0, 8.0, 7.0]})
    mp, up = tmp_path / "m.csv", tmp_path / "u.csv"
    m.to_csv(mp, index=False)
    u.to_csv(up, index=False)
    s = ost.Session(str(mp), auto_sync_workspace=False)
    s.merge(str(up), on="id", update=True)
    s.destring("소득", ignore=",", force=True)
    s.encode("성별", "성별코드")
    s.set_measure("성별코드", "nominal")
    s.rename_var("성별코드", "sex")
    s.decode("sex", "sex_txt")
    s.set_value_labels("z", {1: "낮음", 9: "높음"})
    assert s.value_labels["sex"] == {1: "남", 2: "여"} and s.measure_levels == {"sex": "Nominal"}
    s.save_data(str(tmp_path / "out.dta"))
    r = ost.Session(str(tmp_path / "out.dta"), auto_sync_workspace=False)
    assert r.value_labels["sex"] == {1: "남", 2: "여"} and r.value_labels["z"] == {1: "낮음", 9: "높음"}
    pytest.importorskip("pyreadstat")
    with pytest.warns(UserWarning, match="SPSS"):
        s.save_data(str(tmp_path / "out.sav"))
    r2 = ost.Session(str(tmp_path / "out.sav"), auto_sync_workspace=False)
    assert r2.value_labels["sex"] == {1: "남", 2: "여"} and "v_merge" in r2.df.columns
    script = tmp_path / "replay.py"
    script.write_text(s.to_python_script(dataset_path=str(mp)), encoding="utf-8")
    out = ost.run_script(str(script), sync_workspace=False)
    assert out["df"].astype(str).equals(s.df.astype(str))
    s.undo_data_op()
    assert "z" not in s.value_labels and s.measure_levels == {"sex": "Nominal"}


def test_stata_transpilation_executes(tmp_path):
    assert transpile_stata_line("merge 1:1 id using b, keep(match master) keepus(x y) nogen") == \
        "sess.merge('b.dta', on=['id'], kind='1:1', keep=['match', 'master'], keepusing=['x', 'y'], indicator=False)"
    sem = transpile_stata_line("sem (Visual -> x1 x2 x3) (y <- Visual z), method(mlmv) vce(robust) group(g)")
    assert "Visual =~ x1 + x2 + x3" in sem and "y ~ Visual + z" in sem and "estimator='mlr'" in sem
    assert "invariance='scalar'" in sem                     # Stata ginvariant default mcoef mcons
    assert transpile_stata_line("gsem (u1 u2 <- _cons), logit lclass(C 3)") == "sess.lca(categorical=['u1', 'u2'], classes=3)"
    assert transpile_stata_line("frobnicate x").startswith("raise NotImplementedError")
    lg = pd.DataFrame({"id": [1, 1, 2, 2], "t": [1, 2, 1, 2], "y": [5, 6, 8, 9], "s": ["a", "b", "a", "b"]})
    lg.to_csv(tmp_path / "lg.csv", index=False)
    code = transpile_stata_script(
        "reshape wide y s, i(id) j(t)\n"
        "egen tot = rowtotal(y1 y2)\n"
        "encode s1, gen(s1c)\n"
        'label define ab 1 "A" 2 "B"\n'
        "label values s1c ab\n"
        "collapse (mean) tot, by(s1c)\n")
    sess = ost.Session(str(tmp_path / "lg.csv"), auto_sync_workspace=False)
    exec(code, {"sess": sess})
    assert sess.df.values.tolist() == [[1.0, 14.0]]                    # (5+6 + 8+9) / 2 within s1c == 1
    assert sess.value_labels.get("s1c") == {1: "A", 2: "B"}            # labels follow the by() variable
