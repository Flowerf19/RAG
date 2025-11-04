"""
Chunker Config Loader
====================
Load chunker configuration from YAML file
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ChunkerConfig:
    """Load and manage chunker configuration"""
    
    _config = None  # Singleton cache
    
    @classmethod
    def load(cls, config_path: str = None) -> Dict[str, Any]:
        """
        Load chunker configuration from YAML file
        
        Args:
            config_path: Path to config file (default: config/chunker_config.yaml)
            
        Returns:
            Configuration dictionary
        """
        if cls._config is not None:
            return cls._config
            
        if config_path is None:
            # Default path
            config_path = Path(__file__).parent.parent / "config" / "chunker_config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cls._config = yaml.safe_load(f)
        
        return cls._config
    
    @classmethod
    def get_chunker_config(cls, chunker_type: str = "default") -> Dict[str, Any]:
        """
        Get configuration for specific chunker type
        
        Args:
            chunker_type: Type of chunker (default, fixed_size, semantic, etc.)
            
        Returns:
            Configuration dictionary for that chunker
        """
        config = cls.load()
        
        if chunker_type not in config:
            # Fallback to default
            return config.get("default", {})
        
        return config[chunker_type]


def get_chunker_config(chunker_type: str = "default") -> Dict[str, Any]:
    """
    Convenience function to get chunker config
    
    Args:
        chunker_type: Type of chunker
        
    Returns:
        Configuration dictionary
    """
    return ChunkerConfig.get_chunker_config(chunker_type)
