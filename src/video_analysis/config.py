"""全局配置管理模块。

从 YAML 文件加载默认配置，支持环境变量覆盖。
所有模块通过 `get_config()` 统一获取配置，不依赖 GPU。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# 默认配置文件相对于项目根目录的路径
_DEFAULT_CONFIG_PATH = "configs/default.yaml"

# 环境变量前缀：PSY_VIDEO_<SECTION>_<KEY>
_ENV_PREFIX = "PSY_VIDEO_"


class Config:
    """配置容器，支持点号访问嵌套字典。"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        value = self._data.get(name)
        if value is None:
            raise AttributeError(f"Config key not found: {name}")
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, name: str, default: Any = None) -> Any:
        """安全获取配置值，支持点号分隔的路径如 'thresholds.sedentary.max_displacement_px'。"""
        keys = name.split(".")
        node: Any = self._data
        for key in keys:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return default
            if node is None:
                return default
        return node

    def to_dict(self) -> Dict[str, Any]:
        """返回原始字典副本。"""
        return dict(self._data)


def _find_project_root() -> Path:
    """向上查找项目根目录（包含 configs/ 目录）。"""
    current = Path.cwd()
    for _ in range(10):
        if (current / "configs").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # 回退：环境变量或当前目录
    return Path(os.environ.get("PSY_PROJECT_ROOT", Path.cwd()))


def _apply_env_overrides(data: Dict[str, Any], prefix: str = _ENV_PREFIX) -> None:
    """递归应用环境变量覆盖。环境变量格式: PSY_VIDEO_SECTION_KEY=value。"""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        # PSY_VIDEO_THRESHOLDS_SEDENTARY_MAX_DISPLACEMENT_PX -> thresholds.sedentary.max_displacement_px
        path = env_key[len(prefix):].lower()
        keys = path.split("_")

        # 尝试类型转换
        typed_val: Any = env_val
        if env_val.isdigit():
            typed_val = int(env_val)
        elif env_val.replace(".", "", 1).replace("-", "", 1).isdigit():
            typed_val = float(env_val)
        elif env_val.lower() in ("true", "false"):
            typed_val = env_val.lower() == "true"

        # 导航到嵌套字典的叶子节点
        node: Any = data
        for key in keys[:-1]:
            if key not in node:
                node[key] = {}
            node = node[key]
        node[keys[-1]] = typed_val


def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置。

    Args:
        config_path: YAML 配置文件路径。为 None 时使用默认路径。

    Returns:
        Config 对象。

    Raises:
        FileNotFoundError: 配置文件不存在。
    """
    if config_path is None:
        project_root = _find_project_root()
        config_path = str(project_root / _DEFAULT_CONFIG_PATH)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    _apply_env_overrides(data)
    return Config(data)


# 模块级单例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例。首次调用时自动加载。"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """重置配置单例（主要用于测试）。"""
    global _config
    _config = None
