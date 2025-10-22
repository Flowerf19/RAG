"""
HuggingFace Token Manager
========================
Manages HuggingFace API token loading from multiple sources.
Single Responsibility: Token management and validation.
"""

import os
from pathlib import Path
from typing import Optional


class HuggingFaceTokenManager:
    """
    Manages HuggingFace API token loading from multiple sources.
    
    Sources (in priority order):
    1. Explicit parameter
    2. Environment variables (HF_TOKEN, HUGGINGFACE_TOKEN)
    3. Streamlit secrets
    4. secrets.toml file
    """

    def __init__(self):
        self._token: Optional[str] = None
        self._loaded = False

    def get_token(self, explicit_token: Optional[str] = None) -> Optional[str]:
        """
        Get HuggingFace API token from multiple sources.
        
        Args:
            explicit_token: Token passed explicitly
            
        Returns:
            API token or None if not found
        """
        if not self._loaded:
            self._token = self._load_token(explicit_token)
            self._loaded = True
        
        return self._token

    def _load_token(self, explicit_token: Optional[str] = None) -> Optional[str]:
        """Load token from multiple sources in priority order."""
        
        # 1. Explicit parameter
        if explicit_token:
            return explicit_token
        
        # 2. Environment variables
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            return token
        
        # 3. Streamlit secrets
        token = self._load_from_streamlit_secrets()
        if token:
            return token
        
        # 4. secrets.toml file
        token = self._load_from_secrets_toml()
        if token:
            return token
        
        return None

    def _load_from_streamlit_secrets(self) -> Optional[str]:
        """Load token from Streamlit secrets."""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'huggingface' in st.secrets:
                hf_secrets = st.secrets['huggingface']
                return hf_secrets.get('api_token') or hf_secrets.get('hf_token')
        except ImportError:
            pass  # Streamlit not available
        
        return None

    def _load_from_secrets_toml(self) -> Optional[str]:
        """Load token from secrets.toml file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return None  # TOML not available
        
        # Try multiple possible locations
        possible_paths = [
            Path.cwd() / ".streamlit" / "secrets.toml",
            Path(__file__).parent.parent.parent.parent / ".streamlit" / "secrets.toml",  # From huggingface/
            Path.home() / ".streamlit" / "secrets.toml",
        ]
        
        for secrets_path in possible_paths:
            if secrets_path.exists():
                try:
                    with open(secrets_path, 'rb') as f:
                        secrets = tomllib.load(f)
                        if 'huggingface' in secrets:
                            hf_secrets = secrets['huggingface']
                            token = hf_secrets.get('api_token') or hf_secrets.get('hf_token')
                            if token:
                                return token
                except Exception:
                    continue  # Try next path
        
        return None

    def validate_token(self, token: str) -> bool:
        """
        Validate token format and basic structure.
        
        Args:
            token: Token to validate
            
        Returns:
            True if token looks valid
        """
        if not token or not isinstance(token, str):
            return False
        
        # HF tokens typically start with 'hf_'
        return token.startswith('hf_') and len(token) > 10

    def clear_cache(self):
        """Clear cached token (for testing)."""
        self._token = None
        self._loaded = False


# Global instance for reuse
_token_manager = HuggingFaceTokenManager()

def get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get HuggingFace token.
    
    Args:
        explicit_token: Token passed explicitly
        
    Returns:
        API token or None
    """
    return _token_manager.get_token(explicit_token)