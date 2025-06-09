from dataclasses import dataclass, field
from typing import Type

import torch
import tinycudann as tcnn

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.field_heads import FieldHeadNames


@dataclass
class SemanticNerfModelConfig(NerfactoModelConfig):
    """SemanticNeRF Model Configuration (TCNN field)"""
    _target: Type = field(default_factory=lambda: SemanticNerfModel)

    use_tcnn: bool = True
    hidden_dim: int = 64
    n_layers: int = 2


class TCNNField(Field):
	def __init__(self, encoding, mlp_base):
		super().__init__()
		self.encoding = encoding
		self.mlp_base = mlp_base

		def forward(self, ray_samples, compute_normals: bool = False, **kwargs):
			x = ray_samples.frustums.get_positions()
			x = x.contiguous().to(torch.float32)
			encoded = self.encoding(x)
			out = self.mlp_base(encoded)
			return {"rgb": out[..., :3], "density": out[..., 3:4]}


			self.field = TCNNField(self.encoding, self.mlp_base)


class SemanticNerfModel(NerfactoModel):
    """SemanticNeRF Model with TCNN-based Field"""

    config: SemanticNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.use_tcnn:
            import tinycudann as tcnn

            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 1.5,
                }
            )

            self.mlp_base = tcnn.Network(
                n_input_dims=self.encoding.n_output_dims,
                n_output_dims=4,  # RGB + density
                network_config={
                    "otype": "FullyFusedMLP",
                    "n_neurons": self.config.hidden_dim,
                    "n_hidden_layers": self.config.n_layers,
                    "activation": "ReLU",
                    "output_activation": "None"
                }
            )
