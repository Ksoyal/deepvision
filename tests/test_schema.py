"""离线测试:不依赖 API key,验证数据结构与序列化逻辑。

运行:  python -m pytest tests/  或  python tests/test_schema.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepvision.schema import Scene, Primitive, Relation


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
    sys.exit(1 if failed else 0)
