"""离线测试:不依赖 API key,覆盖从模型输出到结构化 Scene 的全流程。

涵盖:
- schema:坐标计算、JSON 往返、锚点文本、按 id 查询
- vision:JSON 抠取容错、坐标归一化、响应缓存、端到端解析(打桩 HTTP)

运行:  python -m pytest tests/  或  python tests/test_schema.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepvision.schema import Scene, Primitive, Relation
from deepvision import vision
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


def test_anchored_text_has_ids_and_coords():
    text = _sample_scene().to_anchored_text()
    assert "[input_email]" in text
    assert "bbox=(0.100,0.400,0.600,0.460)" in text
    assert "input_email --above--> btn_submit" in text


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
        relations=[], width=800, height=600)
    vision._normalize_coords(s)
    assert _approx(s.by_id("a").box[:2], (0.1, 0.2))


def test_normalize_pixel_scale():
    # 像素标度仅在最大坐标 > 1000 时触发(否则按 0~1000 处理)
    s = Scene(summary="t", primitives=[
        Primitive(id="a", type="point", point=(1500, 1000), label="x")],
        relations=[], width=2000, height=1200)
    vision._normalize_coords(s)
    assert _approx(s.by_id("a").point, (0.75, 1000 / 1200))


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
        # 输入坐标最大 500(≤1000)→ 按 0~1000 标度归一化(各除以 1000)
        assert _approx(p.box[:2], (0.1, 0.2)), p.box
        assert counter[0] == 1, f"应只调 1 次 API,实际 {counter[0]}"
    finally:
        mk.undo()


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


def _tiny_png_bytes(w, h):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


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





