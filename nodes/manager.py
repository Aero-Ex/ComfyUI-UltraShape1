"""Lazy model manager for UltraShape1 (runs inside isolated subprocess)."""

import os
import sys
import gc
import torch
from omegaconf import OmegaConf

from common import ULTRASHAPE_MODELS_DIR, CONFIG_DIR

# Global manager instance persisted in the worker process
_MANAGER = None

def get_manager(config=None):
    """Get the global manager instance, initializing it if needed."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = UltraShapePipelineManager()
    
    # Update manager settings if a new config is provided
    if config:
        _MANAGER.update_config(config)
    
    return _MANAGER

class UltraShapePipelineManager:
    """Manages loading and caching of UltraShape pipelines."""
    
    def __init__(self):
        self.pipeline = None
        self.current_config_hash = None
        self.device = None
        self.dtype = None
        self.config = None

    def update_config(self, config_dict):
        """Update internal config and determine if pipeline needs reload."""
        # Simple hash-like check for config changes
        config_hash = (
            config_dict.get("checkpoint"),
            config_dict.get("config"),
            config_dict.get("dtype"),
            config_dict.get("attention_backend"),
            config_dict.get("low_vram"),
            config_dict.get("disk_offload")
        )
        
        if config_hash != self.current_config_hash:
            print(f"[UltraShape] Config changed, clearing cached pipeline...")
            self.cleanup()
            self.current_config_hash = config_hash
            self.config = config_dict
            
    def get_pipeline(self):
        """Load and return the pipeline based on current config."""
        if self.pipeline is not None:
            return self.pipeline

        if not self.config:
            raise ValueError("Manager has no config set - call update_config first")

        import comfy.model_management as model_management
        from ultrashape.pipelines import UltraShapePipeline
        from ultrashape.utils.misc import instantiate_from_config

        checkpoint = self.config["checkpoint"]
        config_file = self.config["config"]
        dtype_str = self.config["dtype"]
        attention_backend = self.config["attention_backend"]
        low_vram = self.config["low_vram"]
        disk_offload = self.config.get("disk_offload", False)

        # Set attention backend env var
        if attention_backend == "sage_attn":
            os.environ["USE_SAGEATTN"] = "1"
        else:
            os.environ.pop("USE_SAGEATTN", None)

        self.device = model_management.get_torch_device()
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_str]

        ckpt_path = os.path.join(ULTRASHAPE_MODELS_DIR, checkpoint)
        config_path = os.path.join(CONFIG_DIR, config_file)

        print(f"[UltraShape] Loading model {checkpoint} (dtype={dtype_str}, backend={attention_backend})...")
        cfg = OmegaConf.load(config_path)

        # In extreme disk_offload mode, we return a shell pipeline that 
        # loads components on the fly. For now, we reuse the existing 
        # UltraShapePipeline internal logic.
        
        vae = instantiate_from_config(cfg.model.params.vae_config)
        dit = instantiate_from_config(cfg.model.params.dit_cfg)
        conditioner = instantiate_from_config(cfg.model.params.conditioner_config)
        scheduler = instantiate_from_config(cfg.model.params.scheduler_cfg)
        image_processor = instantiate_from_config(cfg.model.params.image_processor_cfg)

        weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        vae.load_state_dict(weights['vae'], strict=True)
        dit.load_state_dict(weights['dit'], strict=True)
        conditioner.load_state_dict(weights['conditioner'], strict=True)

        vae.eval().to(self.device, dtype=self.dtype)
        dit.eval().to(self.device, dtype=self.dtype)
        conditioner.eval().to(self.device, dtype=self.dtype)

        if hasattr(vae, 'enable_flashvdm_decoder'):
            vae.enable_flashvdm_decoder()

        self.pipeline = UltraShapePipeline(
            vae=vae,
            model=dit,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor
        )

        if low_vram:
            self.pipeline.enable_model_cpu_offload()
            
        return self.pipeline

    def cleanup(self):
        """Unload pipeline and free VRAM."""
        if self.pipeline:
            # Attempt to move models to CPU before deleting
            try:
                self.pipeline.vae.cpu()
                self.pipeline.model.cpu()
                self.pipeline.conditioner.cpu()
            except:
                pass
            del self.pipeline
            self.pipeline = None
            
        self.device = None
        self.dtype = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
