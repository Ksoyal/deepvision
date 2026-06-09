"""DeepVision 命令行入口。

用法示例:
  deepvision image.png                      # 结构化解析,输出锚点文本
  deepvision image.png --json               # 输出原始结构化 JSON
  deepvision image.png --refine             # 对低置信区域自动局部细化
  cat img.png | deepvision -                 # 从 stdin 读取
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import os
import subprocess
import sys
from pathlib import Path

from .config import Config
from .vision import describe_structured
from .refine import auto_refine
from .verify import verify_scene


def _read_input(src: str) -> "str | bytes":
    if src == "-":
        return sys.stdin.buffer.read()
    return src  # 文件路径或 URL/data URI 交给下游处理


def _is_windows() -> bool:
    return os.name == "nt"


def _image_to_png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _grab_clipboard_pillow():
    from PIL import ImageGrab

    return ImageGrab.grabclipboard()


def _grab_clipboard_windows() -> bytes | None:
    """Windows fallback: read bitmap clipboard in an STA PowerShell process.

    Pillow can return None for some Windows clipboard bitmap formats. The
    fallback keeps the image in memory and returns PNG bytes without writing a
    temp file.
    """
    if not _is_windows():
        return None

    script = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$img = [System.Windows.Forms.Clipboard]::GetImage()
if ($null -eq $img) {
    exit 2
}
$stream = New-Object System.IO.MemoryStream
try {
    $img.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    [Convert]::ToBase64String($stream.ToArray())
}
finally {
    $stream.Dispose()
    $img.Dispose()
}
"""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    data = next(
        (line.strip() for line in reversed(completed.stdout.splitlines()) if line.strip()),
        "",
    )
    if not data:
        return None
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None


def _grab_clipboard():
    """抓取剪贴板内容,返回 (来源标签, 图片) 列表。

    剪贴板可能是:位图(返回单项)、或复制的图片文件(返回多项路径)。
    """
    obj = _grab_clipboard_pillow()
    if obj is None:
        data = _grab_clipboard_windows()
        if data is not None:
            return [("<剪贴板>", data)]
        raise RuntimeError("剪贴板里没有图片")
    if isinstance(obj, list):
        if not obj:
            raise RuntimeError("剪贴板的文件列表为空")
        return [(str(p), str(p)) for p in obj]  # 路径交给下游按文件处理
    return [("<剪贴板>", _image_to_png_bytes(obj))]


def cmd_init(argv) -> int:
    """生成全局配置文件 ~/.deepvision/config.json(任何目录均可读取)。

    可带参数一步配好,例:
      deepvision init --api-key sk-xxx --base-url https://... --model gpt-4o
    不带参数则生成模板,供手动编辑。
    """
    import json

    ap = argparse.ArgumentParser(prog="deepvision init",
                                 description="生成或写入配置文件")
    ap.add_argument("--api-key", dest="api_key", help="API key")
    ap.add_argument("--base-url", dest="base_url", help="OpenAI 兼容端点")
    ap.add_argument("--model", help="模型 id")
    ap.add_argument("-f", "--force", action="store_true",
                    help="覆盖已存在的配置文件")
    args = ap.parse_args(argv)

    dest = Path.home() / ".deepvision" / "config.json"
    if dest.exists() and not args.force:
        print(f"配置已存在:{dest}", file=sys.stderr)
        print("如需覆盖请加 --force", file=sys.stderr)
        return 1

    cfg = {
        "api_key": args.api_key or "在这里填你的 API key",
        "base_url": args.base_url or "https://api.openai.com/v1",
        "model": args.model or "gpt-4o",
        "temperature": 0.1,
        "max_edge": 1024,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"已生成配置:{dest}")
    if not args.api_key:
        print("下一步:编辑该文件填入 api_key,或重跑 init 带上 --api-key", file=sys.stderr)
    return 0


def cmd_cache(argv) -> int:
    """查看或清空响应缓存。

      deepvision cache            # 显示缓存位置、条目数、占用大小
      deepvision cache --clear    # 清空缓存
    """
    from .vision import cache_stats, cache_clear

    ap = argparse.ArgumentParser(prog="deepvision cache", description="管理响应缓存")
    ap.add_argument("--clear", action="store_true", help="清空全部缓存条目")
    args = ap.parse_args(argv)

    if args.clear:
        removed = cache_clear()
        print(f"已清空缓存,删除 {removed} 条")
        return 0

    s = cache_stats()
    mb = s["bytes"] / (1024 * 1024)
    print(f"缓存目录:{s['dir']}")
    print(f"条目数:{s['entries']}")
    print(f"占用:{mb:.2f} MB")
    return 0


def _process_one(image, args, cfg):
    """处理单张图,返回 scene。"""
    scene = describe_structured(image, cfg)
    if args.verify:
        scene = verify_scene(image, scene, level=args.verify, cfg=cfg)
    if args.refine:
        scene = auto_refine(image, scene, cfg=cfg)
    return scene


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "init":
        return cmd_init(argv[1:])
    if argv and argv[0] == "cache":
        return cmd_cache(argv[1:])

    ap = argparse.ArgumentParser(prog="deepvision", description="结构化视觉解析")
    ap.add_argument("image", nargs="*", default=None,
                    help="图片路径(可传多个批量处理),或 - 表示从 stdin 读取;用 -c 时可省略")
    ap.add_argument("-c", "--clipboard", action="store_true",
                    help="读取剪贴板里的图片")
    ap.add_argument("--json", action="store_true", help="输出原始结构化 JSON")
    ap.add_argument("--refine", action="store_true", help="低置信区域自动局部细化")
    ap.add_argument("--verify", nargs="?", const="suspicious", default=None,
                    choices=["suspicious", "full"],
                    help="核查力度:不写=不核查;--verify=只查可疑(默认);"
                         "--verify full=全量逐个核查(最严格,免费档慢/易限流)")
    ap.add_argument("-m", "--model", help="覆盖模型 id")
    ap.add_argument("-t", "--temperature", type=float, help="采样温度")
    ap.add_argument("-s", "--max-edge", type=int, dest="max_edge", help="长边像素上限")
    ap.add_argument("--no-cache", action="store_true",
                    help="禁用响应缓存,强制重新请求(默认开启缓存,省额度)")
    args = ap.parse_args(argv)

    cfg = Config.load(model=args.model, temperature=args.temperature,
                      max_edge=args.max_edge,
                      cache=False if args.no_cache else None)

    # 收集待处理图片为 (来源标签, 图片) 列表
    items = []
    if args.clipboard:
        try:
            items = _grab_clipboard()
        except Exception as e:  # noqa: BLE001
            print(f"错误[剪贴板]: {e}", file=sys.stderr)
            return 1
    elif args.image:
        items = [(src, _read_input(src)) for src in args.image]
    else:
        ap.error("需要提供图片路径,或用 -c 读取剪贴板")

    json_results = []
    failed = False
    for src, image in items:
        try:
            scene = _process_one(image, args, cfg)
            if args.json:
                json_results.append({
                    "source": src,
                    "scene": scene.to_dict(),
                })
            else:
                if len(items) > 1:
                    print(f"# === {src} ===")
                print(scene.to_anchored_text())
        except Exception as e:  # noqa: BLE001
            print(f"错误[{src}]: {e}", file=sys.stderr)
            failed = True

    if args.json:
        import json
        # 单图时直接输出对象,多图输出数组,兼顾向后兼容
        payload = json_results[0] if len(json_results) == 1 else json_results
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
