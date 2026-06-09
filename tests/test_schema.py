"""离线测试:不依赖 API key,覆盖从模型输出到结构化 Scene 的全流程。

涵盖:
- schema:坐标计算、JSON 往返、锚点文本、按 id 查询
- vision:JSON 抠取容错、坐标归一化、响应缓存、端到端解析(打桩 HTTP)

运行:  python -m pytest tests/  或  python tests/test_schema.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepvision.schema import (
    Scene, Primitive, Composite, Relation,
    derive_geometric_relations, sort_reading_order,
)
from deepvision import cli
from deepvision import vision
import deepvision.refine as refine
import deepvision.verify as verify
from deepvision.config import Config


def _sample_scene() -> Scene:
    return Scene(
        summary="一个登录表单",
        primitives=[
            Primitive(id="input_email", type="bbox", label="邮箱输入框",
                      box=(0.10, 0.40, 0.60, 0.46), text="email"),
            Primitive(id="btn_submit", type="bbox", label="提交按钮",
                      box=(0.12, 0.80, 0.28, 0.86), text="Submit",
                      confidence=0.95),
            Primitive(id="cursor", type="point", label="光标", point=(0.3, 0.43)),
        ],
        relations=[
            Relation(subj="input_email", rel="above", obj="btn_submit"),
        ],
        width=800, height=600,
    )


def _approx(a, b, eps=1e-9):
    return abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps


# ---- schema 层 -------------------------------------------------------------

def test_center_bbox_and_point():
    s = _sample_scene()
    assert _approx(s.by_id("input_email").center(), (0.35, 0.43))
    assert _approx(s.by_id("cursor").center(), (0.3, 0.43))


def test_roundtrip_json():
    s = _sample_scene()
    restored = Scene.from_dict(s.to_dict())
    assert restored.summary == s.summary
    assert len(restored.primitives) == 3
    assert restored.by_id("btn_submit").confidence == 0.95
    assert restored.relations[0].rel == "above"


def test_roundtrip_composites_and_parent_links():
    s = Scene(
        summary="两道题",
        primitives=[
            Primitive(id="num_10", type="bbox", label="题号",
                      box=(0.02, 0.1, 0.08, 0.2), text="10、",
                      role="problem_number", parent="problem_10"),
            Primitive(id="expr_10", type="bbox", label="极限表达式",
                      box=(0.25, 0.1, 0.75, 0.3), parent="problem_10"),
        ],
        composites=[
            Composite(id="problem_10", type="composite", label="第10题",
                      role="problem", box=(0.02, 0.08, 0.75, 0.32),
                      text="求数列的极限 lim_{n→∞} ...",
                      children=["num_10", "expr_10"]),
        ],
    )

    restored = Scene.from_dict(s.to_dict())

    assert restored.by_id("problem_10").role == "problem"
    assert restored.by_id("num_10").parent == "problem_10"
    assert restored.composites[0].children == ["num_10", "expr_10"]


def test_anchored_text_has_ids_and_coords():
    text = _sample_scene().to_anchored_text()
    assert "[input_email]" in text
    assert "bbox=(0.100,0.400,0.600,0.460)" in text
    assert "input_email --above--> btn_submit" in text


def test_anchored_text_lists_composites_before_primitives():
    s = Scene(
        summary="表单",
        primitives=[
            Primitive(id="email_input", type="bbox", label="邮箱输入框",
                      box=(0.1, 0.4, 0.6, 0.5), parent="login_form"),
        ],
        composites=[
            Composite(id="login_form", type="composite", label="登录表单",
                      role="form", box=(0.05, 0.2, 0.7, 0.8),
                      text="邮箱、密码和提交按钮",
                      children=["email_input"]),
        ],
    )

    text = s.to_anchored_text()

    assert "# 语义单元" in text
    assert "[login_form] 登录表单 role=form" in text
    assert 'children=[email_input]' in text
    assert text.index("[login_form]") < text.index("[email_input]")


def test_by_id_missing_returns_none():
    assert _sample_scene().by_id("nope") is None


# ---- vision: JSON 抠取容错 -------------------------------------------------

def test_extract_json_clean():
    assert vision._extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    raw = '```json\n{"a": 1, "b": [2, 3]}\n```'
    assert vision._extract_json(raw) == {"a": 1, "b": [2, 3]}


def test_extract_json_with_prose():
    raw = '这是结果:\n{"a": 1}\n以上就是分析。'
    assert vision._extract_json(raw) == {"a": 1}


def test_extract_json_trailing_comma():
    assert vision._extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_extract_json_braces_in_string():
    raw = '{"label": "a {nested} brace", "n": 1}'
    assert vision._extract_json(raw) == {"label": "a {nested} brace", "n": 1}


def test_extract_json_raises_when_absent():
    try:
        vision._extract_json("没有任何 JSON")
    except ValueError:
        return
    raise AssertionError("应在无 JSON 时抛 ValueError")


# ---- vision: 坐标归一化 ----------------------------------------------------

def test_normalize_already_unit():
    s = _sample_scene()
    vision._normalize_coords(s)
    assert _approx(s.by_id("input_email").box[:2], (0.10, 0.40))


def test_normalize_thousand_scale():
    s = Scene(summary="t", primitives=[
        Primitive(id="a", type="bbox", label="x", box=(100, 200, 500, 460))],
        relations=[])
    vision._normalize_coords(s)
    assert _approx(s.by_id("a").box[:2], (0.1, 0.2))


def test_normalize_pixel_scale():
    # 像素标度仅在最大坐标 > 1000 时触发(否则按 0~1000 处理)
    s = Scene(summary="t", primitives=[
        Primitive(id="a", type="point", point=(1500, 1000), label="x")],
        relations=[], width=2000, height=1200)
    vision._normalize_coords(s)
    assert _approx(s.by_id("a").point, (0.75, 1000 / 1200))


def test_normalize_small_pixel_bbox_uses_image_size():
    """像素坐标即使小于 1000,也应优先按图片宽高归一化。"""
    s = Scene(summary="t", primitives=[
        Primitive(id="a", type="bbox", label="x", box=(100, 200, 500, 460))],
        relations=[], width=800, height=600)
    vision._normalize_coords(s)
    assert _approx(s.by_id("a").box[:2], (100 / 800, 200 / 600))


def test_normalize_composite_bbox_uses_image_size():
    s = Scene(
        summary="t",
        composites=[
            Composite(id="problem_10", type="composite", label="第10题",
                      box=(20, 40, 700, 320)),
        ],
        width=800,
        height=600,
    )

    vision._normalize_coords(s)

    assert _approx(s.by_id("problem_10").box[:2], (20 / 800, 40 / 600))


# ---- vision: 缓存 + 端到端解析(打桩 HTTP,全程离线)----------------------

class _FakeResp:
    status_code = 200
    headers: dict = {}

    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeGetResp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def _patch_api(monkey, content, counter):
    """把 requests.post 换成返回固定 content 的桩,counter 记调用次数。"""
    def fake_post(url, headers=None, json=None, timeout=None):
        counter[0] += 1
        return _FakeResp(content)
    monkey(vision.requests, "post", fake_post)


class _Monkey:
    """极简打桩器:记录并在退出时还原属性,避免依赖 pytest fixture。"""
    def __init__(self):
        self._saved = []

    def set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._saved):
            setattr(obj, name, old)


_FAKE_PROMPT = "system prompt"
_FAKE_JSON = (
    '{"summary": "s", "primitives": ['
    '{"id": "a", "type": "bbox", "label": "x", "box": [100, 200, 500, 460]}'
    '], "relations": []}'
)


def _isolated_cache(mk, tmp):
    """把缓存目录指到临时目录,避免污染真实 ~/.deepvision。"""
    mk.set(vision, "_CACHE_DIR", tmp)


def test_call_api_caches_hit(tmp_path_factory=None):
    import tempfile
    mk = _Monkey()
    counter = [0]
    tmp = Path(tempfile.mkdtemp()) / "cache"
    try:
        _isolated_cache(mk, tmp)
        _patch_api(mk.set, "RESULT", counter)
        cfg = Config(api_key="k", base_url="http://x", model="m", cache=True)
        a = vision._call_api(cfg, _FAKE_PROMPT, "data:img", "q")
        b = vision._call_api(cfg, _FAKE_PROMPT, "data:img", "q")
        assert a == b == "RESULT"
        assert counter[0] == 1, f"应只发 1 次请求,实际 {counter[0]}"
    finally:
        mk.undo()


def test_call_api_no_cache_refetches():
    import tempfile
    mk = _Monkey()
    counter = [0]
    tmp = Path(tempfile.mkdtemp()) / "cache"
    try:
        _isolated_cache(mk, tmp)
        _patch_api(mk.set, "RESULT", counter)
        cfg = Config(api_key="k", base_url="http://x", model="m", cache=False)
        vision._call_api(cfg, _FAKE_PROMPT, "data:img", "q")
        vision._call_api(cfg, _FAKE_PROMPT, "data:img", "q")
        assert counter[0] == 2, f"关缓存应发 2 次,实际 {counter[0]}"
    finally:
        mk.undo()


def test_describe_structured_end_to_end():
    """端到端:打桩 HTTP 返回带像素坐标的 JSON,验证解析+归一化贯通。"""
    import tempfile
    mk = _Monkey()
    counter = [0]
    tmp = Path(tempfile.mkdtemp()) / "cache"
    try:
        _isolated_cache(mk, tmp)
        # 让 _call_api 返回固定 JSON;_to_data_uri 用真实小图,避免再打桩
        mk.set(vision, "_load_prompt", lambda name: _FAKE_PROMPT)
        _patch_api(mk.set, _FAKE_JSON, counter)
        cfg = Config(api_key="k", base_url="http://x", model="m", cache=False)

        img = _tiny_png_bytes(800, 600)
        scene = vision.describe_structured(img, cfg)
        assert scene.summary == "s"
        p = scene.by_id("a")
        assert p is not None and p.type == "bbox"
        # 输入坐标是 800x600 原图像素,应按图像宽高归一化
        assert _approx(p.box[:2], (100 / 800, 200 / 600)), p.box
        assert counter[0] == 1, f"应只调 1 次 API,实际 {counter[0]}"
    finally:
        mk.undo()


def test_describe_structured_uses_detail_prompt():
    """detail 应进入 prompt,从而让缓存和输出粒度可控。"""
    import tempfile
    mk = _Monkey()
    prompts = []
    tmp = Path(tempfile.mkdtemp()) / "cache"
    try:
        _isolated_cache(mk, tmp)
        mk.set(vision, "_load_prompt", lambda name: _FAKE_PROMPT)

        def fake_call(cfg, prompt, uri, question=""):
            prompts.append(prompt)
            return _FAKE_JSON

        mk.set(vision, "_call_api", fake_call)
        cfg = Config(api_key="k", base_url="http://x", model="m", cache=False)
        scene = vision.describe_structured(_tiny_png_bytes(20, 10), cfg, detail="fine")
        assert scene.meta["detail"] == "fine"
    finally:
        mk.undo()

    assert len(prompts) == 1
    assert "detail=fine" in prompts[0]
    assert "精细 OCR" in prompts[0]


def test_describe_structured_rejects_unknown_detail():
    cfg = Config(api_key="k", base_url="http://x", model="m", cache=False)
    try:
        vision.describe_structured(_tiny_png_bytes(10, 10), cfg, detail="dense")
    except ValueError as e:
        assert "detail" in str(e)
        return
    raise AssertionError("未知 detail 不应继续调用视觉模型")


def test_structured_rerun_hits_cache():
    """structured 客观解析:同图重跑应命中缓存,不重复调用 API。"""
    import tempfile
    mk = _Monkey()
    counter = [0]
    tmp = Path(tempfile.mkdtemp()) / "cache"
    try:
        _isolated_cache(mk, tmp)
        mk.set(vision, "_load_prompt", lambda name: _FAKE_PROMPT)
        _patch_api(mk.set, _FAKE_JSON, counter)
        cfg = Config(api_key="k", base_url="http://x", model="m", cache=True)

        img = _tiny_png_bytes(800, 600)
        vision.describe_structured(img, cfg)
        vision.describe_structured(img, cfg)  # 同图重跑
        assert counter[0] == 1, f"同图重跑应命中缓存,实际调 {counter[0]} 次"
    finally:
        mk.undo()


def test_cache_eviction_keeps_limit():
    """写入超过上限时,按最旧淘汰,条目数回到上限。"""
    import tempfile, os
    mk = _Monkey()
    tmp = Path(tempfile.mkdtemp()) / "cache"
    try:
        _isolated_cache(mk, tmp)
        # 先写满,再用显式 mtime 拉开新旧次序(避免同毫秒写入导致顺序不定)
        for i in range(4):
            vision._cache_put(f"key{i}", f"v{i}", max_entries=0)
            os.utime(tmp / f"key{i}.txt", (1000 + i, 1000 + i))
        # 触发淘汰:再写一条、上限 3,应剩 3 条且淘汰最旧的 key0
        vision._cache_put("key4", "v4", max_entries=3)
        os.utime(tmp / "key4.txt", (2000, 2000))
        vision._evict_if_needed(3)
        files = list(tmp.glob("*.txt"))
        assert len(files) == 3, f"应淘汰到 3 条,实际 {len(files)}"
        assert vision._cache_get("key4") == "v4"
        assert vision._cache_get("key0") is None
    finally:
        mk.undo()


def test_cache_eviction_unlimited_when_zero():
    """max_entries=0 表示不限,不淘汰。"""
    import tempfile
    mk = _Monkey()
    tmp = Path(tempfile.mkdtemp()) / "cache"
    try:
        _isolated_cache(mk, tmp)
        for i in range(5):
            vision._cache_put(f"key{i}", f"v{i}", max_entries=0)
        assert len(list(tmp.glob("*.txt"))) == 5
    finally:
        mk.undo()


def test_cache_stats_and_clear():
    import tempfile
    mk = _Monkey()
    tmp = Path(tempfile.mkdtemp()) / "cache"
    try:
        _isolated_cache(mk, tmp)
        vision._cache_put("a", "hello", max_entries=0)
        vision._cache_put("b", "world", max_entries=0)
        s = vision.cache_stats()
        assert s["entries"] == 2 and s["bytes"] == 10
        removed = vision.cache_clear()
        assert removed == 2
        assert vision.cache_stats()["entries"] == 0
    finally:
        mk.undo()


def test_to_data_uri_downloads_http_url():
    """README 承诺支持图片 URL,应下载后转成 data URI。"""
    mk = _Monkey()
    counter = [0]
    payload = _tiny_png_bytes(20, 10)

    def fake_get(url, timeout=None):
        counter[0] += 1
        assert url == "https://example.test/image.png"
        assert timeout is not None
        return _FakeGetResp(payload)

    try:
        mk.set(vision.requests, "get", fake_get)
        data_uri, w, h = vision._to_data_uri("https://example.test/image.png", 1024)
        assert data_uri.startswith("data:image/png;base64,")
        assert (w, h) == (20, 10)
        assert counter[0] == 1
    finally:
        mk.undo()


def test_verify_load_image_accepts_data_uri():
    """verify/refine 这类二阶段能力也应复用主解析支持的图片输入。"""
    import base64

    payload = _tiny_png_bytes(12, 8)
    data_uri = "data:image/png;base64," + base64.b64encode(payload).decode()
    img = verify._load_image(data_uri)
    assert img.size == (12, 8)


def test_refine_region_accepts_data_uri():
    """局部细化应和主解析一样接受 data URI 输入。"""
    import base64

    mk = _Monkey()
    payload = _tiny_png_bytes(12, 8)
    data_uri = "data:image/png;base64," + base64.b64encode(payload).decode()
    prim = Primitive(
        id="panel",
        type="bbox",
        label="面板",
        box=(0.0, 0.0, 1.0, 1.0),
    )
    try:
        mk.set(refine, "_load_prompt", lambda name: _FAKE_PROMPT)
        mk.set(refine, "_call_api", lambda cfg, prompt, uri, question="": _FAKE_JSON)
        cfg = Config(api_key="k", base_url="http://x", model="m", cache=False)
        scene = refine.refine_region(data_uri, prim, cfg=cfg)
        assert scene.meta["refined_from"] == "panel"
    finally:
        mk.undo()


# ---- cli: 友好错误 ---------------------------------------------------------

def test_cli_clipboard_error_is_friendly():
    """剪贴板无图片时不应打印 traceback。"""
    import contextlib
    import io

    mk = _Monkey()
    out = io.StringIO()
    err = io.StringIO()

    try:
        mk.set(cli, "_grab_clipboard",
               lambda: (_ for _ in ()).throw(RuntimeError("剪贴板里没有图片")))
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["-c"])
    finally:
        mk.undo()

    assert code == 1
    assert out.getvalue() == ""
    assert "错误[剪贴板]: 剪贴板里没有图片" in err.getvalue()
    assert "Traceback" not in err.getvalue()


def test_cli_process_one_passes_detail_to_structured_parser():
    class Args:
        detail = "fine"
        verify = None
        refine = False

    seen = []
    mk = _Monkey()
    try:
        mk.set(cli, "describe_structured",
               lambda image, cfg, detail="standard": seen.append(detail) or Scene(summary="ok"))
        scene = cli._process_one(b"img", Args(), Config(api_key="k", base_url="http://x", model="m"))
    finally:
        mk.undo()

    assert scene.summary == "ok"
    assert seen == ["fine"]


def test_grab_clipboard_uses_windows_fallback_when_pillow_misses_bitmap():
    payload = _tiny_png_bytes(10, 6)
    mk = _Monkey()

    try:
        mk.set(cli, "_grab_clipboard_pillow", lambda: None)
        mk.set(cli, "_grab_clipboard_windows", lambda: payload)
        assert cli._grab_clipboard() == [("<剪贴板>", payload)]
    finally:
        mk.undo()


def test_windows_clipboard_fallback_uses_sta_memory_stream():
    import base64

    payload = _tiny_png_bytes(9, 7)
    calls = []

    class Done:
        returncode = 0
        stdout = base64.b64encode(payload).decode() + "\n"
        stderr = ""

    def fake_run(cmd, capture_output=None, text=None, timeout=None, check=None):
        calls.append({
            "cmd": cmd,
            "capture_output": capture_output,
            "text": text,
            "timeout": timeout,
            "check": check,
        })
        return Done()

    mk = _Monkey()
    try:
        mk.set(cli, "_is_windows", lambda: True)
        mk.set(cli.subprocess, "run", fake_run)
        assert cli._grab_clipboard_windows() == payload
    finally:
        mk.undo()

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert "-STA" in cmd
    script = cmd[-1]
    assert "System.IO.MemoryStream" in script
    assert "$env:TEMP" not in script
    assert "clipboard_image" not in script


def _tiny_png_bytes(w, h):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


# ---- schema: 几何关系推导(纯坐标,无需 API)------------------------------

def _rels_set(rels):
    return {(r.subj, r.rel, r.obj) for r in rels}


def test_derive_contains_minimal_parent():
    """contains 只连最小直接容器,不产生传递闭包。"""
    prims = [
        Primitive(id="panel", type="bbox", label="面板", box=(0.0, 0.0, 1.0, 1.0)),
        Primitive(id="card", type="bbox", label="卡片", box=(0.1, 0.1, 0.5, 0.5)),
        Primitive(id="icon", type="bbox", label="图标", box=(0.15, 0.15, 0.2, 0.2)),
    ]
    rels = _rels_set(derive_geometric_relations(prims))
    assert ("card", "contains", "icon") in rels
    assert ("panel", "contains", "card") in rels
    # icon 的直接容器是 card,不应再由 panel 直接 contains
    assert ("panel", "contains", "icon") not in rels


def test_derive_left_of_nearest_in_row():
    """同一行内只连水平最近邻,不连跨过中间元素的远邻。"""
    prims = [
        Primitive(id="a", type="bbox", label="a", box=(0.0, 0.4, 0.1, 0.5)),
        Primitive(id="b", type="bbox", label="b", box=(0.2, 0.4, 0.3, 0.5)),
        Primitive(id="c", type="bbox", label="c", box=(0.4, 0.4, 0.5, 0.5)),
    ]
    rels = _rels_set(derive_geometric_relations(prims))
    assert ("a", "left_of", "b") in rels
    assert ("b", "left_of", "c") in rels
    assert ("a", "left_of", "c") not in rels  # 跨过 b 的远邻不连


def test_derive_above_requires_horizontal_overlap():
    """垂直方向只在水平投影重叠时成立;错位的列不连。"""
    prims = [
        Primitive(id="top", type="bbox", label="上", box=(0.1, 0.0, 0.3, 0.1)),
        Primitive(id="bottom", type="bbox", label="下", box=(0.1, 0.5, 0.3, 0.6)),
        Primitive(id="side", type="bbox", label="旁", box=(0.7, 0.0, 0.9, 0.1)),
    ]
    rels = _rels_set(derive_geometric_relations(prims))
    assert ("top", "above", "bottom") in rels
    assert ("top", "above", "side") not in rels  # 水平不重叠


def test_derive_geometric_relations_do_not_cross_parents():
    """不同父容器的元素不应产生跨块几何关系。"""
    prims = [
        Primitive(id="num_10", type="bbox", label="10", parent="problem_10",
                  box=(0.0, 0.1, 0.1, 0.2)),
        Primitive(id="expr_10", type="bbox", label="表达式10", parent="problem_10",
                  box=(0.2, 0.1, 0.4, 0.2)),
        Primitive(id="num_11", type="bbox", label="11", parent="problem_11",
                  box=(0.0, 0.5, 0.1, 0.6)),
        Primitive(id="expr_11", type="bbox", label="表达式11", parent="problem_11",
                  box=(0.2, 0.5, 0.4, 0.6)),
    ]

    rels = _rels_set(derive_geometric_relations(prims))

    assert ("num_10", "left_of", "expr_10") in rels
    assert ("num_11", "left_of", "expr_11") in rels
    assert ("num_10", "above", "num_11") not in rels
    assert ("expr_10", "above", "expr_11") not in rels


def test_derive_filters_low_value_formula_token_relations():
    """公式 token 内部的水平顺序关系噪声应被过滤。"""
    prims = [
        Primitive(id="p_lim", type="bbox", label="极限符号",
                  role="formula_part", parent="formula_1",
                  box=(0.10, 0.10, 0.18, 0.18), text="lim"),
        Primitive(id="p_lpar", type="bbox", label="左括号",
                  role="formula_part", parent="formula_1",
                  box=(0.20, 0.10, 0.22, 0.20), text="("),
        Primitive(id="p_rpar", type="bbox", label="右括号",
                  role="formula_part", parent="formula_1",
                  box=(0.40, 0.10, 0.42, 0.20), text=")"),
        Primitive(id="p_exp", type="bbox", label="指数",
                  role="formula_part", parent="formula_1",
                  box=(0.44, 0.08, 0.50, 0.14), text="n"),
    ]

    rels = _rels_set(derive_geometric_relations(prims))

    assert not any(rel == "left_of" for _, rel, _ in rels)


def test_derive_keeps_fraction_structure_relations():
    """涉及分数线的垂直关系是结构信息,应保留。"""
    prims = [
        Primitive(id="num", type="bbox", label="分子",
                  role="formula_part", parent="fraction_1",
                  box=(0.30, 0.10, 0.50, 0.16), text="a+b"),
        Primitive(id="bar", type="bbox", label="分数线",
                  role="formula_part", parent="fraction_1",
                  box=(0.30, 0.17, 0.50, 0.18)),
        Primitive(id="den", type="bbox", label="分母",
                  role="formula_part", parent="fraction_1",
                  box=(0.30, 0.19, 0.50, 0.25), text="c+d"),
    ]

    rels = _rels_set(derive_geometric_relations(prims))

    assert ("num", "above", "bar") in rels
    assert ("bar", "above", "den") in rels
    assert not any(rel == "left_of" for _, rel, _ in rels)


def test_derive_point_not_a_container():
    """退化成点的基元没有面积,不该当容器。"""
    prims = [
        Primitive(id="pt", type="point", label="点", point=(0.5, 0.5)),
        Primitive(id="box", type="bbox", label="框", box=(0.4, 0.4, 0.6, 0.6)),
    ]
    rels = derive_geometric_relations(prims)
    assert not any(r.subj == "pt" and r.rel == "contains" for r in rels)


# ---- schema: 阅读顺序排序 --------------------------------------------------

def test_sort_reading_order_rows_then_cols():
    prims = [
        Primitive(id="r2c1", type="bbox", label="", box=(0.0, 0.8, 0.1, 0.9)),
        Primitive(id="r1c2", type="bbox", label="", box=(0.5, 0.1, 0.6, 0.2)),
        Primitive(id="r1c1", type="bbox", label="", box=(0.0, 0.1, 0.1, 0.2)),
    ]
    ordered = [p.id for p in sort_reading_order(prims)]
    assert ordered == ["r1c1", "r1c2", "r2c1"]


def test_sort_reading_order_unlocated_sink_to_end():
    prims = [
        Primitive(id="ghost", type="bbox", label=""),  # 无坐标
        Primitive(id="real", type="bbox", label="", box=(0.0, 0.1, 0.1, 0.2)),
    ]
    ordered = [p.id for p in sort_reading_order(prims)]
    assert ordered == ["real", "ghost"]


# ---- config: 配置优先级与占位校验 ----------------------------------------

def test_config_project_file_overrides_global_file():
    import json
    import tempfile
    from deepvision import config as config_module

    mk = _Monkey()
    tmp = Path(tempfile.mkdtemp())
    global_cfg = tmp / "home.json"
    project_cfg = tmp / ".deepvision.json"
    global_cfg.write_text(json.dumps({
        "api_key": "global-key",
        "base_url": "https://global.example/v1",
        "model": "global-model",
        "temperature": 0.9,
    }), encoding="utf-8")
    project_cfg.write_text(json.dumps({
        "base_url": "https://project.example/v1",
        "model": "project-model",
    }), encoding="utf-8")

    try:
        mk.set(config_module, "CONFIG_PATHS", [global_cfg, project_cfg])
        cfg = Config.load()
        assert cfg.api_key == "global-key"
        assert cfg.base_url == "https://project.example/v1"
        assert cfg.model == "project-model"
        assert cfg.temperature == 0.9
    finally:
        mk.undo()


def test_config_openai_env_replaces_placeholder_key():
    import json
    import os
    import tempfile
    from deepvision import config as config_module

    mk = _Monkey()
    tmp = Path(tempfile.mkdtemp())
    cfg_file = tmp / "config.json"
    cfg_file.write_text(json.dumps({
        "api_key": "在这里填你的 API key",
        "base_url": "https://api.example/v1",
        "model": "vision-model",
    }), encoding="utf-8")
    old_openai_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "env-openai-key"

    try:
        mk.set(config_module, "CONFIG_PATHS", [cfg_file])
        cfg = Config.load()
        assert cfg.api_key == "env-openai-key"
    finally:
        if old_openai_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_openai_key
        mk.undo()


def test_config_placeholder_api_key_is_not_ready():
    cfg = Config(
        api_key="在这里填你的 API key",
        base_url="https://api.example/v1",
        model="vision-model",
    )
    try:
        cfg.require_ready()
    except RuntimeError as e:
        assert "api_key" in str(e)
        return
    raise AssertionError("占位 api_key 不应通过配置就绪校验")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
