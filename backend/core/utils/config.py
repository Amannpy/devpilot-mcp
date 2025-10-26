"""
Config redirect - all config is now in backend.config
This file exists only for backwards compatibility
"""
from backend.config import settings, app_config, AppConfig, config

# Re-export everything for backwards compatibility
__all__ = ['settings', 'app_config', 'AppConfig', 'config', 'Config']

# Alias
Config = AppConfig