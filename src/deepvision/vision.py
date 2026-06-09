"""核心视觉解析:图像 -> 结构化 Scene。

describe_structured() 把图片解析成带坐标锚点的 Scene(本项目的核心价值):
每个元素带稳定 id、归一化坐标与空间关系,供下游文本模型精确引用。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import requests
from PIL import Image

from .config import Config
from .schema import Scene, derive_geometric_relations, sort_reading_order


_PROMPT_DIR = Path(__file__).parent / "prompts"
_CACHE_DIR = Path.home() / ".deepvision" / "cache"
_IMAGE_FETCH_TIMEOUT = 30


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
    vals = []
    for p in scene.primitives:
        if p.box:
            vals.extend(p.box)
        if p.point:
            vals.extend(p.point)
    if not vals:
        return scene
    mx = max(vals)
    if mx <= 1.5:
        return scene  # 已是归一化坐标

    def _fits_image_pixels() -> bool:
        if not scene.width or not scene.height:
            return False
        for p in scene.primitives:
            if p.box:
                x0, y0, x1, y1 = p.box
                if max(x0, x1) > scene.width or max(y0, y1) > scene.height:
                    return False
            if p.point:
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

    for p in scene.primitives:
        if p.box:
            x0, y0, x1, y1 = p.box
            p.box = (_clamp(x0 / sx), _clamp(y0 / sy),
                     _clamp(x1 / sx), _clamp(y1 / sy))
        if p.point:
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


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        return img.convert("RGB")


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
                        cfg: Optional[Config] = None) -> Scene:
    """核心接口:把图片解析成带坐标锚点的结构化 Scene。

    产出的是客观全量解析:同一张图无论下游想问什么,都引用这同一份
    结构化表示(回答交给下游文本模型)。因此不向视觉模型传任何问题——
    既保证解析不被问题带偏,也让缓存能跨问题复用(同图同模型同 prompt
    即命中)。
    """
    cfg = cfg or Config.load()
    data_uri, w, h = _to_data_uri(image, cfg.max_edge)
    prompt = _load_prompt("structured.md")
    raw = _call_api(cfg, prompt, data_uri)
    data = _extract_json(raw)
    scene = Scene.from_dict(data)
    scene.width, scene.height = w, h
    scene = _normalize_coords(scene)
    # 坐标就绪后:按阅读顺序聚簇,并用坐标补齐几何关系(模型只给语义关系)。
    scene.primitives = sort_reading_order(scene.primitives)
    scene.relations = scene.relations + derive_geometric_relations(scene.primitives)
    return scene
