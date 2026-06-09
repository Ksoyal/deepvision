"""配置管理。

优先级:显式传参 > 环境变量 > 项目配置(./.deepvision.json) > 全局配置(~/.deepvision/config.json)。
兼容 OpenAI 风格的多模态接口(base_url + api_key + model)。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CONFIG_PATHS = [
    Path.home() / ".deepvision" / "config.json",
    Path.cwd() / ".deepvision.json",
]

PLACEHOLDERS = {
    "api_key": {"在这里填你的 API key", "<key>", "<api_key>", "sk-xxx"},
    "base_url": {"<端点>", "<base_url>"},
    "model": {"<模型id>", "<model>"},
}


@dataclass
class Config:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.1
    max_edge: int = 1024  # 长边缩放上限,控制 token 成本
    timeout: int = 120
    max_retries: int = 4  # 429 限流时的指数退避重试次数
    cache: bool = True  # 缓存 API 响应,同图同请求不重复调用(省额度)
    cache_max_entries: int = 1000  # 缓存条目上限,超出按最近最少使用淘汰;0 表示不限

    @classmethod
    def load(cls, **overrides) -> "Config":
        data: dict = {}
        for p in CONFIG_PATHS:
            if p.is_file():
                try:
                    data.update(json.loads(p.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    pass

        env_map = {
            "api_key": "DEEPVISION_API_KEY",
            "base_url": "DEEPVISION_BASE_URL",
            "model": "DEEPVISION_MODEL",
        }
        for field_name, env in env_map.items():
            val = os.environ.get(env)
            if val:
                data[field_name] = val

        # OpenAI 标准变量兜底:只在 api_key 为空或仍是模板占位时生效
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and cls._is_missing("api_key", data.get("api_key")):
            data["api_key"] = openai_key

        # 显式 overrides 最高优先级
        for k, v in overrides.items():
            if v is not None:
                data[k] = v

        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @staticmethod
    def _is_missing(name: str, value: object) -> bool:
        if value is None:
            return True
        text = str(value).strip()
        if not text:
            return True
        return text in PLACEHOLDERS.get(name, set())

    def require_ready(self) -> None:
        """请求前校验必填项均已配置,缺失则明确报错(不偷偷使用默认端点)。"""
        missing = [name for name in ("api_key", "base_url", "model")
                   if self._is_missing(name, getattr(self, name))]
        if missing:
            raise RuntimeError(
                f"配置缺失:{', '.join(missing)}。"
                "运行 `deepvision init --api-key ... --base-url ... --model ...`,"
                "或设置环境变量 DEEPVISION_API_KEY / DEEPVISION_BASE_URL / DEEPVISION_MODEL。"
            )

    # 向后兼容别名
    require_key = require_ready
