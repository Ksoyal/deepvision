"""视觉证据自检:逐基元裁剪复核,剔除幻觉元素。

解决免费小模型"自信幻觉"的问题:首轮解析可能凭常识先验编造不存在
的元素(且给高 confidence)。本模块把每个基元对应区域单独裁出来,
用 verify.md 提示反问模型"这块区域真有这个东西吗",仅保留有视觉
证据支持的基元,并同步清理悬空关系。

这是论文 point-while-reasoning "回头核对"思想的工程落地。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from PIL import Image

from .config import Config
from .schema import Scene, Primitive
from .vision import _to_data_uri, _call_api, _extract_json, _load_prompt
from .refine import _crop_bbox


def _load_image(image: Union[str, Path, bytes]) -> Image.Image:
    if isinstance(image, (str, Path)) and Path(image).is_file():
        return Image.open(image).convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    raise ValueError("verify 需要文件路径或 bytes 输入")


def observe_region(img: Image.Image, p: Primitive,
                   cfg: Config) -> Dict[str, Any]:
    """无暗示地观察基元区域,返回 {seen_text, description, is_blank}。

    关键:不把基元的 label/text 告诉模型,避免诱导确认(否则弱模型会
    顺着暗示把不存在的东西"看"出来)。只问"你实际看到了什么"。
    """
    if p.type == "bbox" and p.box:
        crop = _crop_bbox(img, p.box, pad=0.08)
    elif p.point:
        x, y = p.point
        box = (max(0, x - 0.08), max(0, y - 0.08),
               min(1, x + 0.08), min(1, y + 0.08))
        crop = _crop_bbox(img, box, pad=0.0)
    else:
        return {"seen_text": "", "description": "无坐标", "is_blank": True}

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    data_uri, _, _ = _to_data_uri(buf.getvalue(), cfg.max_edge)

    prompt = _load_prompt("verify.md")  # 无暗示版,不含 label/text
    raw = _call_api(cfg, prompt, data_uri)
    try:
        return _extract_json(raw)
    except Exception:  # noqa: BLE001 — 解析失败时保守(当作非空白)
        return {"seen_text": "", "description": raw[:80], "is_blank": False}


def _text_matches(claimed: str, seen: str) -> bool:
    """轻量比对:声称的文字是否在实际看到的文字里出现(忽略大小写/空白)。"""
    if not claimed:
        return True  # 没声称文字,不以文字为据
    c = "".join(claimed.lower().split())
    s = "".join(seen.lower().split())
    if not s:
        return False  # 声称有文字却什么都没看到 -> 不符(空串是任意串子串,需显式排除)
    return c in s or s in c


def _suspicion_reason(p: Primitive, conf_threshold: float) -> Optional[str]:
    """纯代码判定基元是否"可疑"(零 API 成本)。返回可疑原因或 None。

    只有可疑基元才值得花一次 API 去裁剪核查,避免在限流档位全量核查。
    信号:低置信度、退化/越界框、极端长宽比。
    """
    if p.confidence is not None and p.confidence < conf_threshold:
        return f"置信度低({p.confidence})"
    if p.box:
        x0, y0, x1, y1 = p.box
        if not (0 <= x0 <= 1 and 0 <= y0 <= 1 and 0 <= x1 <= 1 and 0 <= y1 <= 1):
            return "坐标越界"
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return "退化框(宽或高<=0)"
        if w * h < 0.0004:  # 面积小于全图万分之四,极可能是误标的小点
            return f"框面积过小({w * h:.5f})"
        ratio = max(w, h) / max(min(w, h), 1e-6)
        if ratio > 50:  # 极端细长
            return f"长宽比异常({ratio:.0f})"
    return None


def verify_scene(
    image: Union[str, Path, bytes],
    scene: Scene,
    level: str = "suspicious",
    conf_threshold: float = 0.6,
    cfg: Optional[Config] = None,
) -> Scene:
    """按用户选择的力度核查基元,剔除幻觉元素及悬空关系。

    力度档位(level):
      - "suspicious"(默认):只核查结构可疑的基元(低置信/退化框/越界/
        异常长宽比)。请求数少,避开限流档位,适合幻觉较少的强模型。
      - "full":逐个核查全部基元。最严格,能抓"坐标正常但内容是幻觉"的
        元素;但请求数 = 基元数,免费档容易被 429,慢。

    可疑基元的核查逻辑(基于无暗示观察,不受首轮 label 诱导):
      - 区域被独立描述为空白 -> 剔除
      - 声称有文字但实际看到的对不上 -> 剔除
    全过程记入 meta['verification'] 供审计。
    """
    if level not in ("suspicious", "full"):
        raise ValueError(f"未知核查力度: {level}(可选 suspicious / full)")
    cfg = cfg or Config.load()
    img = _load_image(image)

    kept: List[Primitive] = []
    audit: List[Dict[str, Any]] = []
    for p in scene.primitives:
        # full 档强制核查每个基元;suspicious 档只查命中可疑规则的
        susp = "全量核查" if level == "full" else _suspicion_reason(p, conf_threshold)
        if susp is None:
            kept.append(p)  # 不可疑,直接信任,不花 API
            audit.append({"id": p.id, "label": p.label,
                          "suspicious": False, "rejected": False})
            continue

        obs = observe_region(img, p, cfg)
        seen_text = obs.get("seen_text", "") or ""
        is_blank = bool(obs.get("is_blank"))

        reason = None
        if is_blank:
            reason = "区域被独立描述为空白/仅背景"
        elif p.text and not _text_matches(p.text, seen_text):
            reason = f"声称文字「{p.text}」与实际看到「{seen_text}」不符"

        rejected = reason is not None
        audit.append({
            "id": p.id, "label": p.label, "suspicious": True,
            "suspicion": susp, "claimed_text": p.text, "seen_text": seen_text,
            "description": obs.get("description"),
            "rejected": rejected, "reason": reason,
        })
        if not rejected:
            kept.append(p)

    kept_ids = {p.id for p in kept}
    kept_rels = [r for r in scene.relations
                 if r.subj in kept_ids and r.obj in kept_ids]
    dropped = len(scene.primitives) - len(kept)

    scene.primitives = kept
    scene.relations = kept_rels
    scene.meta["verification"] = {
        "checked": len(audit),
        "dropped": dropped,
        "audit": audit,
    }
    return scene
