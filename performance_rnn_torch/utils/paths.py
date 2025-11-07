"""
Centralized path configuration for the Performance RNN project.

This module provides a single source of truth for all paths used throughout the project,
with support for environment variable overrides and automatic directory creation.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the root directory of the project.

    Returns:
        Path: Absolute path to the project root directory.
    """
    # This file is in performance_rnn_torch/utils/paths.py
    # So we go up two levels to get to the project root
    return Path(__file__).parent.parent.parent.resolve()


def get_package_root() -> Path:
    """
    Get the root directory of the performance_rnn_torch package.

    Returns:
        Path: Absolute path to the package root directory.
    """
    # This file is in performance_rnn_torch/utils/paths.py
    # So we go up one level to get to the package root
    return Path(__file__).parent.parent.resolve()


class ProjectPaths:
    """
    Manages all paths used in the project with environment variable support.

    All paths can be overridden using environment variables with the prefix
    'PERFORMANCE_RNN_' followed by the uppercase attribute name.

    Example:
        export PERFORMANCE_RNN_DATA_DIR=/custom/data/path
    """

    def __init__(self):
        self._project_root = get_project_root()
        self._package_root = get_package_root()

    @property
    def project_root(self) -> Path:
        """Root directory of the project."""
        return self._project_root

    @property
    def package_root(self) -> Path:
        """Root directory of the performance_rnn_torch package."""
        return self._package_root

    def _get_path(self, env_var: str, default: Path) -> Path:
        """
        Get a path from environment variable or use default.

        Args:
            env_var: Name of the environment variable
            default: Default path if environment variable is not set

        Returns:
            Path: The resolved path
        """
        env_path = os.getenv(env_var)
        if env_path:
            return Path(env_path).resolve()
        return default

    @property
    def data_dir(self) -> Path:
        """Directory for all data files."""
        return self._get_path('PERFORMANCE_RNN_DATA_DIR', self._project_root / 'data')

    @property
    def midi_dir(self) -> Path:
        """Directory for raw MIDI files."""
        path = self._get_path('PERFORMANCE_RNN_MIDI_DIR', self.data_dir / 'midi')
        return path

    @property
    def processed_dir(self) -> Path:
        """Directory for preprocessed data files."""
        path = self._get_path('PERFORMANCE_RNN_PROCESSED_DIR', self.data_dir / 'processed')
        return path

    @property
    def maestro_dir(self) -> Path:
        """Directory for Maestro dataset."""
        return self._get_path('PERFORMANCE_RNN_MAESTRO_DIR', self.data_dir / 'maestro-v3.0.0')

    @property
    def models_dir(self) -> Path:
        """Directory for saved model checkpoints."""
        path = self._get_path('PERFORMANCE_RNN_MODELS_DIR', self._project_root / 'models')
        return path

    @property
    def output_dir(self) -> Path:
        """Directory for generated MIDI output."""
        path = self._get_path('PERFORMANCE_RNN_OUTPUT_DIR', self._project_root / 'output')
        return path

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        path = self._get_path('PERFORMANCE_RNN_LOGS_DIR', self._project_root / 'logs')
        return path

    @property
    def runs_dir(self) -> Path:
        """Directory for TensorBoard runs."""
        path = self._get_path('PERFORMANCE_RNN_RUNS_DIR', self._project_root / 'runs')
        return path

    @property
    def scripts_dir(self) -> Path:
        """Directory for dataset download scripts."""
        return self.data_dir / 'scripts'

    def ensure_directories(self, create_data_dirs: bool = True) -> None:
        """
        Create all necessary directories if they don't exist.

        Args:
            create_data_dirs: Whether to create data directories (midi, processed, etc.)
                             Set to False if you want to download/prepare data separately.
        """
        # Always create these directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        if create_data_dirs:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.midi_dir.mkdir(parents=True, exist_ok=True)
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            self.scripts_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, session_name: str) -> Path:
        """
        Get the path for a model checkpoint file.

        Args:
            session_name: Name of the model session (e.g., 'train', 'debussy')

        Returns:
            Path: Full path to the model checkpoint file
        """
        if not session_name.endswith('.sess'):
            session_name += '.sess'
        return self.models_dir / session_name

    def get_log_path(self, log_name: Optional[str] = None) -> Path:
        """
        Get the path for a log file.

        Args:
            log_name: Name of the log file (optional, will use timestamp if not provided)

        Returns:
            Path: Full path to the log file
        """
        if log_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = f'model_info_{timestamp}.log'

        if not log_name.endswith('.log'):
            log_name += '.log'

        return self.logs_dir / log_name


# Global instance for easy access
paths = ProjectPaths()


# Convenience functions for backward compatibility
def get_data_dir() -> Path:
    """Get the data directory path."""
    return paths.data_dir


def get_models_dir() -> Path:
    """Get the models directory path."""
    return paths.models_dir


def get_output_dir() -> Path:
    """Get the output directory path."""
    return paths.output_dir
