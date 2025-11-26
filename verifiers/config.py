"""
Configuration management for the verification system.

Supports loading configuration from:
- YAML files
- Python dictionaries
- Environment variables
- Command-line arguments
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from .base import ConfigurationError


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "math": {
        "timeout": 5.0,  # Increased from 2.0 - symbolic computation can be slow
        "symbolic_first": True,
        "numeric_tolerance": 1e-6,
        "enable_units": True,
        "unit_tolerance": 1e-4,
        "max_expression_size": 10000,  # Prevent DoS from huge expressions
        "latex_parsing": True,
    },
    "code": {
        "timeout": 10.0,  # Increased from 5.0 - complex code needs more time
        "docker_image": "nexus-code-verifier:latest",
        "container_pool_size": 5,
        "memory_limit": "512m",
        "cpu_limit": 2.0,
        "network_disabled": True,
        "languages": ["python", "javascript", "bash", "sql"],
        "enable_test_cases": True,
        "max_output_size": 10000,  # Max characters in output
    },
    "tool": {
        "timeout": 5.0,  # Increased from 1.0 - tool validation needs more time
        "schema_validation": "strict",  # 'strict', 'lenient', 'none'
        "mock_execution": False,
        "validate_tool_exists": True,
        "validate_parameter_types": True,
        "max_json_size": 100000,  # Max characters in JSON
    },
}


# ============================================================================
# Configuration Loader
# ============================================================================

class VerifierConfig:
    """
    Configuration manager for verifiers.

    Handles loading, merging, and accessing configuration from multiple sources.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Initial configuration dictionary (optional)
        """
        self._config = DEFAULT_CONFIG.copy()
        if config_dict:
            self.merge(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VerifierConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            VerifierConfig instance

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        path = Path(yaml_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")

        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

        return cls(config_dict)

    @classmethod
    def from_env(cls, prefix: str = "VERIFIER_") -> "VerifierConfig":
        """
        Load configuration from environment variables.

        Environment variable format:
            VERIFIER_MATH_TIMEOUT=2.5
            VERIFIER_CODE_DOCKER_IMAGE=custom:latest
            VERIFIER_TOOL_SCHEMA_VALIDATION=lenient

        Args:
            prefix: Prefix for environment variables (default: "VERIFIER_")

        Returns:
            VerifierConfig instance with values from environment
        """
        config_dict: Dict[str, Any] = {}

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Remove prefix and split into components
            # VERIFIER_MATH_TIMEOUT -> ['math', 'timeout']
            parts = key[len(prefix):].lower().split('_')

            if len(parts) < 2:
                continue

            verifier_type = parts[0]
            setting_name = '_'.join(parts[1:])

            # Create nested dict structure
            if verifier_type not in config_dict:
                config_dict[verifier_type] = {}

            # Try to parse value as appropriate type
            parsed_value = cls._parse_env_value(value)
            config_dict[verifier_type][setting_name] = parsed_value

        return cls(config_dict)

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse environment variable value to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Parsed value (int, float, bool, list, or str)
        """
        # Boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # List (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]

        # String (default)
        return value

    def merge(self, other: Dict[str, Any]):
        """
        Merge another configuration dictionary into this one.

        Args:
            other: Dictionary to merge (overrides existing values)
        """
        for verifier_type, settings in other.items():
            if verifier_type not in self._config:
                self._config[verifier_type] = {}

            if isinstance(settings, dict):
                self._config[verifier_type].update(settings)
            else:
                self._config[verifier_type] = settings

    def get(self, verifier_type: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            verifier_type: Type of verifier ('math', 'code', 'tool')
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        return self._config.get(verifier_type, {}).get(key, default)

    def get_verifier_config(self, verifier_type: str) -> Dict[str, Any]:
        """
        Get all configuration for a specific verifier type.

        Args:
            verifier_type: Type of verifier ('math', 'code', 'tool')

        Returns:
            Dictionary of configuration settings
        """
        return self._config.get(verifier_type, {}).copy()

    def set(self, verifier_type: str, key: str, value: Any):
        """
        Set a configuration value.

        Args:
            verifier_type: Type of verifier ('math', 'code', 'tool')
            key: Configuration key
            value: Configuration value
        """
        if verifier_type not in self._config:
            self._config[verifier_type] = {}
        self._config[verifier_type][key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()

    def save_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file

        Raises:
            ConfigurationError: If file cannot be written
        """
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def validate(self):
        """
        Validate configuration values.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate timeout values
        for verifier_type in ['math', 'code', 'tool']:
            if verifier_type in self._config:
                timeout = self._config[verifier_type].get('timeout')
                if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
                    raise ConfigurationError(
                        f"{verifier_type}.timeout must be positive number, got {timeout}"
                    )

        # Validate math config
        if 'math' in self._config:
            tol = self._config['math'].get('numeric_tolerance')
            if tol is not None and (not isinstance(tol, (int, float)) or tol < 0):
                raise ConfigurationError(f"math.numeric_tolerance must be non-negative, got {tol}")

        # Validate code config
        if 'code' in self._config:
            pool_size = self._config['code'].get('container_pool_size')
            if pool_size is not None and (not isinstance(pool_size, int) or pool_size < 0):
                raise ConfigurationError(f"code.container_pool_size must be non-negative int, got {pool_size}")

            cpu_limit = self._config['code'].get('cpu_limit')
            if cpu_limit is not None and (not isinstance(cpu_limit, (int, float)) or cpu_limit <= 0):
                raise ConfigurationError(f"code.cpu_limit must be positive number, got {cpu_limit}")

        # Validate tool config
        if 'tool' in self._config:
            validation = self._config['tool'].get('schema_validation')
            if validation is not None and validation not in ['strict', 'lenient', 'none']:
                raise ConfigurationError(
                    f"tool.schema_validation must be 'strict', 'lenient', or 'none', got {validation}"
                )

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"VerifierConfig({self._config})"


# ============================================================================
# Convenience Functions
# ============================================================================

def load_config(
    yaml_path: Optional[str] = None,
    env_prefix: str = "VERIFIER_",
    extra_config: Optional[Dict[str, Any]] = None
) -> VerifierConfig:
    """
    Load configuration from multiple sources with priority:
    1. Default configuration (lowest priority)
    2. YAML file (if provided)
    3. Environment variables (if prefix matches)
    4. Extra configuration dict (highest priority)

    Args:
        yaml_path: Optional path to YAML configuration file
        env_prefix: Prefix for environment variables (default: "VERIFIER_")
        extra_config: Optional dictionary to merge last (highest priority)

    Returns:
        VerifierConfig instance with merged configuration

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Start with defaults
    config = VerifierConfig()

    # Load from YAML if provided
    if yaml_path:
        yaml_config = VerifierConfig.from_yaml(yaml_path)
        config.merge(yaml_config.to_dict())

    # Load from environment variables
    env_config = VerifierConfig.from_env(env_prefix)
    config.merge(env_config.to_dict())

    # Merge extra config (highest priority)
    if extra_config:
        config.merge(extra_config)

    # Validate final configuration
    config.validate()

    return config
