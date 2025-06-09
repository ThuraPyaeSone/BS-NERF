import torch
import tinycudann as tcnn
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.field_heads import FieldHeadNames


class SemanticNerfField(Field):
    def __init__(self, aabb, num_images, config):
        super().__init__()
        self.aabb = aabb
        self.num_images = num_images
        self.config = config

        # TCNN HashGrid Encoding
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

        # FullyFused MLP
        self.mlp_base = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=4,  # RGB (3) + Density (1)
            network_config={
                "otype": "FullyFusedMLP",
                "n_neurons": config.hidden_dim,
                "n_hidden_layers": config.n_layers,
                "activation": "ReLU",
                "output_activation": "None"
            }
        )

    def forward(self, ray_samples, compute_normals: bool = False, **kwargs):
        x = ray_samples.frustums.get_positions()
        x = x.contiguous().to(torch.float32)
        encoded = self.encoding(x)
        out = self.mlp_base(encoded)
        return {"rgb": out[..., :3], "density": out[..., 3:4]}