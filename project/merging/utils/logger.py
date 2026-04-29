import logging
import json
from pathlib import Path
from datetime import datetime
import sys

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PipelineLogger:
    def __init__(self, config, log_name="pipeline"):
        self.config = config
        self.log_name = log_name
        self.logger = self._setup_logger()
        self.wandb_enabled = config.get("logging", {}).get("use_wandb", False) and WANDB_AVAILABLE
        
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
            output_dir = Path(self.config.get("output", {}).get("output_dir", "./output"))
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_wandb(self):
        config_dict = {
            "block1": self.config.get("block1_query_processing", {}),
            "block2": self.config.get("block2_retrieval_mode", {}),
            "block3": self.config.get("block3_ranking_mode", {}),
        }
        wandb.init(
            project=self.config.get("logging", {}).get("wandb_project", "nlp-merging"),
            entity=self.config.get("logging", {}).get("wandb_entity", None),
            config=config_dict,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
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
