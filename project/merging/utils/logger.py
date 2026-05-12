import logging
import json
from pathlib import Path
from datetime import datetime
import sys
import os
import importlib
import subprocess

try:
    from wandb.errors.errors import AuthenticationError
except Exception:
    AuthenticationError = Exception

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class PipelineLogger:
    def __init__(self, config, log_name="pipeline"):
        global WANDB_AVAILABLE, wandb
        self.config = config
        self.log_name = log_name
        self.run_id = self._allocate_run_id()
        self.logger = self._setup_logger()

        # Decision to enable wandb: user config and availability.
        requested = bool(config.get("logging", {}).get("use_wandb", False))

        # If requested but wandb not installed, try to install it.
        if requested and not WANDB_AVAILABLE:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb>=0.12.0"])
                wandb = importlib.import_module("wandb")
                WANDB_AVAILABLE = True
            except Exception:
                self.logger.warning("WandB requested but could not be installed/imported. Continuing without WandB.")

        # If an API key is provided via config or environment, attempt to login.
        api_key = config.get("logging", {}).get("wandb_api_key") or os.environ.get("WANDB_API_KEY")
        if api_key and WANDB_AVAILABLE:
            try:
                wandb.login(key=api_key)
            except Exception:
                self.logger.warning("WandB login failed with provided API key.")

        self.wandb_enabled = requested and WANDB_AVAILABLE
        if self.wandb_enabled:
            self._init_wandb()
    
    def _setup_logger(self):
        log_level = self.config.get("logging", {}).get("log_level", "INFO")
        logger = logging.getLogger(self.log_name)
        logger.setLevel(getattr(logging, log_level))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if self.config.get("logging", {}).get("log_file", True):
            output_dir = self._resolve_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / f"pipeline_{self.run_id}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    def _resolve_output_dir(self):
        config_dir = Path(self.config.get("_config_dir", Path.cwd())).resolve()
        output_dir = Path(self.config.get("output", {}).get("output_dir", "./output"))
        return output_dir if output_dir.is_absolute() else (config_dir / output_dir).resolve()

    def _allocate_run_id(self):
        output_dir = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate descriptive run ID from block modes
        block1_mode = self.config.get("block1_query_processing", {}).get("expansion_mode", "none")
        block1_reduce = self.config.get("block1_query_processing", {}).get("reduction_enabled", False)
        block2_mode = self.config.get("block2_retrieval_mode", {}).get("retrieval_type", "tfidf")
        block3_mode = self.config.get("block3_ranking_mode", {}).get("ranking_type", "tfidf")
        
        # Build run ID with optional reduction flag
        reduce_suffix = "_red" if block1_reduce else ""
        run_id_base = f"merge_{block1_mode}_{block2_mode}_{block3_mode}{reduce_suffix}"
        
        # Check for existing runs with this configuration and add counter if needed
        existing_counters = []
        for log_file in output_dir.glob(f"pipeline_{run_id_base}*.log"):
            suffix = log_file.stem.replace(f"pipeline_{run_id_base}", "", 1)
            if suffix == "":
                existing_counters.append(0)
            elif suffix.startswith("_") and suffix[1:].isdigit():
                existing_counters.append(int(suffix[1:]))
        
        if existing_counters:
            counter = max(existing_counters) + 1
            return f"{run_id_base}_{counter}"
        else:
            return run_id_base
    
    def _init_wandb(self):
        config_dict = {
            "block1": self.config.get("block1_query_processing", {}),
            "block2": self.config.get("block2_retrieval_mode", {}),
            "block3": self.config.get("block3_ranking_mode", {}),
            "dataset": self.config.get("dataset", {}),
            "output": self.config.get("output", {}),
            "logging": self.config.get("logging", {}),
        }
        init_kwargs = {
            "project": self.config.get("logging", {}).get("wandb_project", "nlp-merging"),
            "entity": self.config.get("logging", {}).get("wandb_entity", None),
            "config": config_dict,
            "name": self.run_id,
            "mode": "online"
        }

        try:
            wandb.init(**init_kwargs)
            wandb.define_metric("k")
            wandb.define_metric("*", step_metric="k")
        except Exception as e:
            self.logger.error(f"WandB initialization failed: {e}. Continuing without WandB.")
            self.wandb_enabled = False
    
    def log_metrics(self, metrics_dict, step=None):
        self.logger.info(f"Metrics: {metrics_dict}")
        if self.wandb_enabled:
            wandb.log(metrics_dict, step=step)
    
    def log_artifact(self, file_path, artifact_name=None):
        if self.wandb_enabled:
            wandb.save(str(file_path))
            if artifact_name:
                artifact = wandb.Artifact(artifact_name, type='result')
                artifact.add_file(str(file_path))
                wandb.log_artifact(artifact)
    
    def log_plot(self, figure, plot_name):
        if self.wandb_enabled:
            wandb.log({plot_name: figure})
    
    def info(self, msg):
        self.logger.info(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def close(self):
        if self.wandb_enabled:
            wandb.finish()
