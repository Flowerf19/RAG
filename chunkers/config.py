import os
import yaml
from typing import Any, Dict, Optional

class YAMLConfigLoader:
    """YAML config loader cho chunking module."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Ưu tiên rag/config/chunking.yaml
            base = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(base, 'config', 'chunking.yaml')
        self.config_path = os.path.abspath(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            if not isinstance(config, dict):
                raise ValueError(f"Config file {self.config_path} must contain a dictionary, got {type(config)}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key."""
        return self._config.get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section."""
        value = self._config.get(section, {})
        if not isinstance(value, dict):
            return {}
        return value

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested config value."""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def get_all(self) -> Dict[str, Any]:
        """Get all config."""
        return self._config.copy()


class ConfigManager:
    """Singleton Config Manager cho chunking - thread-safe."""
    
    _instance: Optional['ConfigManager'] = None
    _config_loader: Optional[YAMLConfigLoader] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, config_path: Optional[str] = None):
        """Initialize config manager với config path."""
        if cls._instance is None:
            cls._instance = cls()
        cls._config_loader = YAMLConfigLoader(config_path)
    
    @classmethod
    def get_loader(cls) -> YAMLConfigLoader:
        """Get config loader instance."""
        if cls._config_loader is None:
            cls.initialize()
        # At this point, _config_loader should not be None, but add a check to satisfy type checker
        if cls._config_loader is None:
            raise RuntimeError("Config loader could not be initialized.")
        return cls._config_loader
    
    @classmethod
    def get_config(cls, *keys: str, default: Any = None) -> Any:
        """Get config value by nested keys."""
        loader = cls.get_loader()
        if not keys:
            return loader.get_all()
        return loader.get_nested(*keys, default=default)
    
    @classmethod
    def reload(cls, config_path: Optional[str] = None):
        """Reload config from file."""
        cls.initialize(config_path)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to load config."""
    ConfigManager.initialize(config_path)
    return ConfigManager.get_config()


def get_chunking_config(strategy: Optional[str] = None) -> Dict[str, Any]:
    """Get chunking config for specific strategy."""
    config = ConfigManager.get_config("chunking", default={})
    if strategy:
        return config.get(strategy, {})
    return config
