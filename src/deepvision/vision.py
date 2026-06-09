"""核心视觉解析:图像 -> 结构化 Scene。

describe_structured() 把图片解析成带坐标锚点的 Scene(本项目的核心价值):
每个元素带稳定 id、归一化坐标与空间关系,供下游文本模型精确引用。
"""

from __future__ import annotations

import base64
import colorsys
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import requests
from PIL import Image

from .config import Config
from .schema import Scene, derive_geometric_relations, sort_reading_order


_PROMPT_DIR = Path(__file__).parent / "prompts"
_CACHE_DIR = Path.home() / ".deepvision" / "cache"
_IMAGE_FETCH_TIMEOUT = 30
_DETAIL_MODES = {"brief", "standard", "fine"}
_INTENT_MODES = {"general", "count", "ocr", "locate", "inspect"}
_DETAIL_PROMPT_ADDENDA = {
    "brief": """
# 输出粒度(detail=brief)

只输出低噪声概览:保留主要语义单元、主要对象/区域和关键可读文本。
不要拆到单字符、单 token 或细碎装饰元素。适合用户只需要快速看懂图片内容。
""",
    "standard": """
# 输出粒度(detail=standard)

这是默认模式。目标是让没有视觉能力的文本模型看懂图片,同时控制上下文噪声。
- 优先输出语义单元(composites)和少量必要视觉基元(primitives)。
- 不要把普通文本、公式或代码拆成单字符/单 token 元素。
- 公式保留到 expression、fraction、numerator、denominator、exponent、subscript 这类结构层级即可;完整内容写入 composite.text。
- 不要仅为了拼出 OCR 文本而创建字符级 children;children 只引用真实输出的 id。
- 表格可输出 table/row/cell 层级,cell.text 尽量完整;除非文字本身需要定位,不要拆 cell 内字符。
""",
    "fine": """
# 输出粒度(detail=fine)

精细 OCR / 精确结构化模式。用于用户明确要求逐字识别、校对、坐标定位、
表格逐格提取、代码截图精确转写,或 standard 结果不足的场景。
- 可以在必要时输出词级、字符级、代码 token 级或更细的 primitives。
- 仍然必须优先提供 composites,让下游先读整体结构再读细节。
- 每个细粒度 primitive 必须有稳定 id、坐标、text,并用 parent 指向最近的 composite。
- children 只引用真实输出的 id;不要输出无意义背景或装饰碎片。
""",
}
_INTENT_PROMPT_ADDENDA = {
    "count": """
# 任务意图(intent=count)

目标是精确计数。对用户可能关心的重复对象、标记点、独立实例逐个枚举,
再给出总数。不要用常识默认数量替代图像证据,尤其要注意多出、缺失、重叠或异常的可见实例。
计数时先建立候选实例列表,按位置顺序给目标实例编号(target_1、target_2...),再尝试命名;不要先套用标准类别。
即使没有视觉标记,也必须基于图像本身逐个寻找候选实例。
候选实例列表至少检查目标外观、边界、遮挡、重叠、局部可见、相似干扰、异常多出或缺失。
如果标准类别数量不足以覆盖可见实例,保留额外编号实例,不要把它合并或忽略。
视觉标记只是辅助证据。如果存在彩色点、圈注、编号、箭头或其他视觉标记,先判断它们是否与目标实例对应;
若对应,把它们作为计数线索并说明每个实例的位置;若不对应,不要采用标记数量。
遮挡或不确定的实例要标低 confidence 或说明不确定,不要强行并入常规数量。
""",
    "ocr": """
# 任务意图(intent=ocr)

目标是精确转写可读文字。保持自然阅读顺序,尽量完整保留文字、数字、符号、换行、
表格单元格文本和代码片段。对不确定字符标低 confidence,不要按语义补写看不清的内容。
""",
    "locate": """
# 任务意图(intent=locate)

目标是精确定位。优先输出可被引用的 bbox/point、稳定 id、parent 和必要空间关系,
使下游可以回答在哪里、左/右/上/下、第几个、靠近什么等问题。
""",
    "inspect": """
# 任务意图(intent=inspect)

目标是仔细检查、复核或发现异常。逐项列出可疑、不一致、额外、缺失、遮挡或与常识不同的视觉证据。
不要用常识覆盖图像中的异常现象;不确定时明确标注不确定。
""",
}
_TARGET_PROMPT_ADDENDA = {
    "general": """
# 目标对象(target={target})

当前任务关注目标是: {target}。在保持全图客观结构的前提下,优先保留与该目标相关的语义单元、
视觉基元、文字和空间关系。不要因为有 target 就忽略周边必要上下文。
""",
    "count": """
# 目标对象(target={target})

当前任务关注目标是: {target}。只围绕该目标建立可核验的候选实例列表和总数。
不要把其他可数对象混入目标计数;也不要用目标的常识默认数量覆盖图像证据。
即使没有视觉标记,也必须依据目标外观、边界、遮挡、重叠、局部可见部分和相似干扰物逐个核查。
视觉标记只是辅助证据。如果目标附近存在彩色点、圈注、编号、箭头或其他视觉标记,
应先判断它们是否与目标实例对应,再作为候选实例的重要证据。
对目标实例使用从左到右、从上到下或自然阅读顺序的编号,不要因为无法命名为常规类别就丢弃实例。
""",
    "ocr": """
# 目标对象(target={target})

当前任务关注目标是: {target}。完整转写目标区域中的可读内容,并保留自然阅读顺序、换行、
标点、大小写、表格行列、代码缩进和数学结构。遇到公式时要保留分式、括号、根号、
上下标、极限下标、指数、求和/积分上下限等结构关系,不要按语义补写看不清的字符。
目标周边如果包含题号、标题、单位、图例或必要条件,也应作为上下文保留。
""",
    "locate": """
# 目标对象(target={target})

当前任务关注目标是: {target}。优先输出该目标及其关键参照物的稳定 id、bbox/point、parent
和空间关系,使下游可以回答位置、方向、第几个、靠近什么、包含于哪个区域等问题。
不要把与定位无关的细碎文本或装饰元素作为主要输出。
""",
    "inspect": """
# 目标对象(target={target})

当前任务关注目标是: {target}。围绕该目标仔细复核可疑、异常、缺失、额外、遮挡、
重叠或不一致的视觉证据。保留必要上下文,但不要用常识覆盖图像中的异常现象。
不确定时标低 confidence 或明确说明不确定。
""",
}


def _cache_key(model: str, prompt: str, data_uri: str, question: str) -> str:
    """API 响应的缓存键:模型 + 提示 + 图片 + 问题的内容哈希。"""
    h = hashlib.sha256()
    for part in (model, prompt, question, data_uri):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(key: str) -> Optional[str]:
    f = _CACHE_DIR / f"{key}.txt"
    if f.is_file():
        try:
            value = f.read_text(encoding="utf-8")
            f.touch()  # 更新访问时间,LRU 保活:常用条目不被淘汰
            return value
        except OSError:
            return None
    return None


def _evict_if_needed(max_entries: int) -> None:
    """按最近最少使用(mtime)淘汰超出上限的缓存条目。max_entries<=0 表示不限。"""
    if max_entries <= 0:
        return
    try:
        files = list(_CACHE_DIR.glob("*.txt"))
        if len(files) <= max_entries:
            return
        files.sort(key=lambda p: p.stat().st_mtime)  # 最旧的在前
        for p in files[: len(files) - max_entries]:
            p.unlink(missing_ok=True)
    except OSError:
        pass  # 回收失败不影响主流程


def _cache_put(key: str, value: str, max_entries: int = 0) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # 原子写:先写临时文件再替换,避免并发/中断留下半截文件
        tmp = _CACHE_DIR / f"{key}.tmp"
        tmp.write_text(value, encoding="utf-8")
        os.replace(tmp, _CACHE_DIR / f"{key}.txt")
    except OSError:
        return  # 缓存写失败不影响主流程
    _evict_if_needed(max_entries)


def cache_stats() -> dict:
    """返回缓存目录的位置、条目数、总字节数。"""
    entries, total = 0, 0
    try:
        for p in _CACHE_DIR.glob("*.txt"):
            entries += 1
            total += p.stat().st_size
    except OSError:
        pass
    return {"dir": str(_CACHE_DIR), "entries": entries, "bytes": total}


def cache_clear() -> int:
    """清空缓存目录,返回删除的条目数。"""
    removed = 0
    try:
        for p in _CACHE_DIR.glob("*.txt"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed



def _normalize_coords(scene: Scene) -> Scene:
    """把模型可能返回的非 [0,1] 坐标统一压回 [0,1]。

    模型常无视归一化约定,返回三种标度之一:
      - [0,1]:        已正确,不动
      - 0~1000:       Qwen-VL / Gemini 系常见,除以 1000
      - 真实像素:      除以图片宽高
    用"出现过的最大坐标值"来判定标度,避免逐元素误判。
    """
    elements = [*scene.primitives, *scene.composites]
    vals = []
    for p in elements:
        if getattr(p, "box", None):
            vals.extend(p.box)
        if getattr(p, "point", None):
            vals.extend(p.point)
    if not vals:
        return scene
    mx = max(vals)
    if mx <= 1.5:
        return scene  # 已是归一化坐标

    def _fits_image_pixels() -> bool:
        if not scene.width or not scene.height:
            return False
        for p in elements:
            if getattr(p, "box", None):
                x0, y0, x1, y1 = p.box
                if max(x0, x1) > scene.width or max(y0, y1) > scene.height:
                    return False
            if getattr(p, "point", None):
                x, y = p.point
                if x > scene.width or y > scene.height:
                    return False
        return True

    if _fits_image_pixels():
        sx = float(scene.width)
        sy = float(scene.height)
        source = "image_pixels"
    elif mx <= 1000:
        sx = sy = 1000.0  # 0~1000 标度
        source = "thousand_grid"
    else:
        sx = float(scene.width or mx)   # 像素标度
        sy = float(scene.height or mx)
        source = "large_pixels"

    def _clamp(v):
        return min(1.0, max(0.0, v))

    for p in elements:
        if getattr(p, "box", None):
            x0, y0, x1, y1 = p.box
            p.box = (_clamp(x0 / sx), _clamp(y0 / sy),
                     _clamp(x1 / sx), _clamp(y1 / sy))
        if getattr(p, "point", None):
            x, y = p.point
            p.point = (_clamp(x / sx), _clamp(y / sy))
    scene.meta["coord_rescaled"] = {
        "detected_max": mx,
        "scale_x": sx,
        "scale_y": sy,
        "source": source,
    }
    return scene



def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def _normalize_detail(detail: Optional[str]) -> str:
    value = (detail or "standard").strip().lower()
    if value not in _DETAIL_MODES:
        allowed = ", ".join(sorted(_DETAIL_MODES))
        raise ValueError(f"detail 必须是以下之一: {allowed}")
    return value


def _normalize_intent(intent: Optional[str]) -> str:
    value = (intent or "general").strip().lower()
    if value not in _INTENT_MODES:
        allowed = ", ".join(sorted(_INTENT_MODES))
        raise ValueError(f"intent 必须是以下之一: {allowed}")
    return value


def _normalize_target(target: Optional[str]) -> str:
    return (target or "").strip()


def _detail_for_intent(detail: Optional[str], intent: Optional[str]) -> str:
    value = _normalize_detail(detail)
    intent_value = _normalize_intent(intent)
    if intent_value != "general":
        return "fine"
    return value


def _prompt_for_detail(detail: Optional[str]) -> str:
    return _prompt_for_options(detail, "general")


def _prompt_for_options(detail: Optional[str],
                        intent: Optional[str],
                        target: Optional[str] = "",
                        count_hints: Optional[dict] = None) -> str:
    intent_value = _normalize_intent(intent)
    detail_value = _detail_for_intent(detail, intent_value)
    target_value = _normalize_target(target)
    prompt = (
        _load_prompt("structured.md").rstrip()
        + "\n\n"
        + _DETAIL_PROMPT_ADDENDA[detail_value].strip()
        + "\n"
    )
    if intent_value != "general":
        prompt += "\n" + _INTENT_PROMPT_ADDENDA[intent_value].strip() + "\n"
    if target_value:
        prompt += "\n" + _TARGET_PROMPT_ADDENDA[intent_value].format(target=target_value).strip() + "\n"
    hint_text = _format_count_hints(count_hints or {})
    if hint_text:
        prompt += "\n" + hint_text + "\n"
    return prompt


def _format_count_hints(count_hints: dict) -> str:
    markers = count_hints.get("visual_markers") or []
    if not markers:
        return ""
    points = ", ".join(
        f"{m['id']}:{m.get('color', 'color')}@({m['point'][0]:.3f},{m['point'][1]:.3f})"
        for m in markers[:30]
    )
    return (
        "# 本地视觉计数线索\n\n"
        f"visual_markers count={len(markers)} points=[{points}]\n"
        "这些线索来自本地像素检测,表示可能的彩色标记点或注释点,不是语言常识或最终答案。"
        "若目标实例与这些标记点对应,应逐个核对这些标记点;若不对应,不要强行采用该数量。"
    )


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        return img.convert("RGB")


def _image_from_data_uri(data_uri: str) -> Image.Image:
    _, b64 = data_uri.split(",", 1)
    return _image_from_bytes(base64.b64decode(b64))


def _load_image(image: Union[str, Path, bytes]) -> Image.Image:
    """把文件路径、URL、bytes 或 data URI 统一载入为 RGB Image。"""
    if isinstance(image, bytes):
        return _image_from_bytes(image)

    if isinstance(image, str) and image.startswith("data:"):
        try:
            _, b64 = image.split(",", 1)
        except ValueError as e:
            raise ValueError("不支持的 data URI:缺少逗号分隔的 base64 内容") from e
        return _image_from_bytes(base64.b64decode(b64))

    if isinstance(image, str) and _is_http_url(image):
        resp = requests.get(image, timeout=_IMAGE_FETCH_TIMEOUT)
        resp.raise_for_status()
        return _image_from_bytes(resp.content)

    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.is_file():
            with Image.open(path) as img:
                return img.convert("RGB")

    raise ValueError("不支持的图像输入:需要文件路径、URL、bytes 或 data URI")


def _to_data_uri(image: Union[str, Path, bytes], max_edge: int) -> tuple[str, int, int]:
    """把各种输入统一成缩放后的 PNG data URI,返回 (uri, width, height)。"""
    img = _load_image(image)
    w, h = img.size
    scale = min(1.0, max_edge / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, img.width, img.height


def _count_hints_from_data_uri(data_uri: str) -> dict:
    img = _image_from_data_uri(data_uri)
    markers = _detect_visual_markers(img)
    return {"visual_markers": markers} if markers else {}


def _detect_visual_markers(img: Image.Image) -> list[dict[str, Any]]:
    """Detect repeated small saturated marker-like blobs as count evidence."""
    w, h = img.size
    pix = img.load()
    mask: dict[tuple[int, int], str] = {}
    for y in range(h):
        for x in range(w):
            r, g, b = pix[x, y]
            color = _marker_color_name(r, g, b)
            if color:
                mask[(x, y)] = color

    seen = set()
    candidates = []
    max_diameter = max(12, int(min(w, h) * 0.08))
    min_area = max(8, int(w * h * 0.00001))
    max_area = max(200, int(w * h * 0.003))

    for start in list(mask.keys()):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        xs, ys = [], []
        colors: dict[str, int] = {}
        while stack:
            x, y = stack.pop()
            xs.append(x)
            ys.append(y)
            color = mask[(x, y)]
            colors[color] = colors.get(color, 0) + 1
            for nx in (x - 1, x, x + 1):
                for ny in (y - 1, y, y + 1):
                    p = (nx, ny)
                    if p in mask and p not in seen:
                        seen.add(p)
                        stack.append(p)

        area = len(xs)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if not (min_area <= area <= max_area):
            continue
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        if bw > max_diameter or bh > max_diameter:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        fill_ratio = area / max(1, bw * bh)
        if aspect > 1.8 or fill_ratio < 0.35:
            continue
        color = max(colors, key=colors.get)
        candidates.append({
            "area": area,
            "color": color,
            "box": (x0 / w, y0 / h, x1 / w, y1 / h),
            "point": ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h),
        })

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        grouped.setdefault(item["color"], []).append(item)

    markers = []
    for items in grouped.values():
        if len(items) >= 2:
            markers.extend(items)

    markers.sort(key=lambda item: (item["point"][1], item["point"][0]))
    for i, item in enumerate(markers, 1):
        item["id"] = f"visual_marker_{i}"
    return markers


def _marker_color_name(r: int, g: int, b: int) -> str:
    value = max(r, g, b)
    chroma = value - min(r, g, b)
    if value < 140 or chroma < 80:
        return ""
    hue = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0] * 360
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 75:
        return "yellow"
    if hue < 165:
        return "green"
    if hue < 205:
        return "cyan"
    if hue < 265:
        return "blue"
    return "purple"


def _call_api(cfg: Config, prompt: str, data_uri: str, question: str = "") -> str:
    """调用 OpenAI 兼容的多模态 chat 接口。

    内容排布:系统提示在前、用户问题居中、图片在后,有利于 API
    前缀缓存命中(同一张图配不同问题时复用前缀)。
    """
    cfg.require_key()
    if getattr(cfg, "cache", False):
        key = _cache_key(cfg.model, prompt, data_uri, question)
        hit = _cache_get(key)
        if hit is not None:
            return hit

    user_content = []
    if question:
        user_content.append({"type": "text", "text": question})
    user_content.append({"type": "image_url", "image_url": {"url": data_uri}})

    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
    }
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    # 免费档常返回 429,做指数退避重试(尊重 Retry-After 头)
    last_exc = None
    for attempt in range(cfg.max_retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
        if resp.status_code == 429 and attempt < cfg.max_retries:
            ra = resp.headers.get("Retry-After")
            wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 2 ** attempt
            time.sleep(min(wait, 30))
            last_exc = requests.HTTPError(f"429 (第 {attempt + 1} 次,退避 {wait}s)")
            continue
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if getattr(cfg, "cache", False):
            _cache_put(_cache_key(cfg.model, prompt, data_uri, question), content,
                       getattr(cfg, "cache_max_entries", 0))
        return content
    raise last_exc or RuntimeError("API 调用失败")


def _strip_fences(text: str) -> str:
    """去掉 ```json ... ``` 围栏,返回栏内内容(无围栏则原样返回)。"""
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text


def _balanced_json(text: str) -> Optional[str]:
    """从首个 '{' 起按括号深度扫描出第一段平衡的 JSON 对象。

    比 find('{') + rfind('}') 稳健:能跳过字符串内的括号与转义,
    也不会被 JSON 之后夹带的解释文字干扰。
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> dict:
    """从模型输出里稳健地抠出 JSON。

    依次尝试:直接解析 -> 去围栏后解析 -> 平衡括号截取后解析 ->
    去尾随逗号后解析。覆盖免费模型常见的格式瑕疵(围栏、赘述、尾逗号)。
    """
    candidates = []
    stripped = _strip_fences(text.strip())
    candidates.append(stripped)
    balanced = _balanced_json(stripped)
    if balanced:
        candidates.append(balanced)

    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
            # 容错:去掉对象/数组里的尾随逗号 ( ,} 或 ,] ) 再试一次
            fixed = re.sub(r",(\s*[}\]])", r"\1", cand)
            if fixed != cand:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError as e2:
                    last_err = e2
    raise ValueError(f"无法从模型输出解析出 JSON:{last_err}")



def describe_structured(image: Union[str, Path, bytes],
                        cfg: Optional[Config] = None,
                        detail: str = "standard",
                        intent: str = "general",
                        target: str = "") -> Scene:
    """核心接口:把图片解析成带坐标锚点的结构化 Scene。

    默认产出客观全量解析,不传用户问题;intent 只传递任务类型,
    target 只传递短目标词,并按 intent 解释为计数、OCR、定位或复核的关注对象。
    """
    cfg = cfg or Config.load()
    intent = _normalize_intent(intent)
    detail = _detail_for_intent(detail, intent)
    target = _normalize_target(target)
    data_uri, w, h = _to_data_uri(image, cfg.max_edge)
    count_hints = _count_hints_from_data_uri(data_uri) if intent == "count" else {}
    prompt = _prompt_for_options(detail, intent, target, count_hints)
    raw = _call_api(cfg, prompt, data_uri)
    data = _extract_json(raw)
    scene = Scene.from_dict(data)
    scene.width, scene.height = w, h
    scene.meta["detail"] = detail
    scene.meta["intent"] = intent
    if target:
        scene.meta["target"] = target
    if count_hints:
        scene.meta["count_hints"] = count_hints
    scene = _normalize_coords(scene)
    # 坐标就绪后:按阅读顺序聚簇,并用坐标补齐几何关系(模型只给语义关系)。
    scene.primitives = sort_reading_order(scene.primitives)
    scene.relations = scene.relations + derive_geometric_relations(scene.primitives)
    return scene
