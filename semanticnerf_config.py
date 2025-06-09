"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from method_semanticnerf.semanticnerf_datamanager import (
    SemanticNerfDataManagerConfig,
)
from method_semanticnerf.semanticnerf_model import SemanticNerfModelConfig
from method_semanticnerf.semanticnerf_pipeline import (
    SemanticNerfPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


semanticnerf = MethodSpecification(
    config=TrainerConfig(
        method_name="method-semanticnerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=31000,
        mixed_precision=True,
        pipeline=SemanticNerfPipelineConfig(
            datamanager=SemanticNerfDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_res_scale_factor=0.5,  # ↓ Lower image resolution to reduce GPU usage
            ),
            model=SemanticNerfModelConfig(
                eval_num_rays_per_chunk=1 << 11,  # ↓ Safer batch size for 6GB GPU (2048 rays)
                average_init_density=0.01,
                num_nerf_samples_per_ray=32,  # ↓ Fewer samples per ray = less memory
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=50000
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 11),  # ↓ Viewer uses same safe chunk size
        vis="viewer",
    ),
    description="Nerfstudio semantic nerf template.",
)
