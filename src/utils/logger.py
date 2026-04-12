"""
Logging utilities for experiment tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "encRec",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "encRec") -> logging.Logger:
    """Get an existing logger or create a basic one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger


class ExperimentLogger:
    """
    Structured experiment logging with support for metrics, configs, and artifacts.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb

        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logger
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(self.exp_dir / "experiment.log")
        )

        # Initialize wandb if requested
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "encRec",
                    name=f"{experiment_name}_{timestamp}",
                    dir=str(self.exp_dir)
                )
            except ImportError:
                self.logger.warning("wandb not installed, skipping wandb logging")

        self.step = 0
        self.metrics_history = []

    def log_config(self, config: dict):
        """Log experiment configuration."""
        import json
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info(f"Config saved to {config_path}")

        if self.wandb_run:
            import wandb
            wandb.config.update(config)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics for current step."""
        if step is not None:
            self.step = step

        metrics_with_step = {"step": self.step, **metrics}
        self.metrics_history.append(metrics_with_step)

        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in metrics.items()])
        self.logger.info(f"Step {self.step}: {metrics_str}")

        # Log to wandb
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=self.step)

        self.step += 1

    def log_model(self, model, name: str = "model"):
        """Save model checkpoint."""
        import torch
        model_path = self.exp_dir / f"{name}.pt"
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

    def finish(self):
        """Finalize logging."""
        import json
        # Save metrics history
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        if self.wandb_run:
            import wandb
            wandb.finish()

        self.logger.info(f"Experiment finished. Logs saved to {self.exp_dir}")
