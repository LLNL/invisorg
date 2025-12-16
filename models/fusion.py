"""
Model Class for the Microbiome Maps Fusion Model.
"""

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------- Classes ---------------------------------------------------------
#
# region Fusion Model
class MmFusionModel(nn.Module):
    """
    Fusion model that ensembles ViT and CNN backbones, averages (or weighted averages) their feature embeddings,
    and performs late fusion for final classification.
    """

    def __init__(self, vit_models, cnn_models, num_classes=1, output_dim=512, dropout_rate=0.2, use_weighted=False,
                 vit_weight=0.5, is_multi_label: bool = False, use_hfe: bool = False, hfe_dim: int = None):
        """
        Args:
            vit_models: list of ViT models (mlp stripped to Identity)
            cnn_models: list of CNN models (classifier removed, just conv_blocks remain)
            num_classes: number of output classes
            output_dim: size to project ViT and CNN features to
            dropout_rate: dropout probability in fusion head
            use_weighted: whether to use weighted ViT/CNN fusion
            vit_weight: weight assigned to ViT feature (only if use_weighted=True)
        """
        super().__init__()

        self.is_multi_label = is_multi_label
        self.use_hfe = use_hfe

        assert len(vit_models) > 0, "Need at least one ViT model"
        assert len(cnn_models) > 0, "Need at least one CNN model"

        self.vit_models    = nn.ModuleList(vit_models)
        self.cnn_models    = nn.ModuleList(cnn_models)
        self.use_weighted  = use_weighted
        self.vit_weight    = vit_weight

        # Extract hidden dims
        self.vit_hidden_d    = vit_models[0].hidden_d
        self.cnn_flatten_dim = cnn_models[0].flatten_dim

        self.v_proj = nn.Linear(self.vit_hidden_d, output_dim)
        self.c_proj = nn.Linear(self.cnn_flatten_dim, output_dim)

        # HFE projection layer
        if self.use_hfe:
            assert hfe_dim is not None, "hfe_dim must be provided when use_hfe is True"
            self.hfe_proj = nn.Linear(hfe_dim, output_dim)

        # Update classifier input dimension
        fusion_input_dim = output_dim * 2
        if self.use_hfe:
            fusion_input_dim = output_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, num_classes)
        )


    def forward(self, x, hfe_features=None):
        batch_size = x.size(0)

        # Forward through all ViTs.
        vit_features = torch.stack([vit(x) for vit in self.vit_models], dim=0)  # (n_vits, batch, hidden_d)

        # Forward pass through CNNs.
        cnn_features = []
        for cnn in self.cnn_models:
            features = cnn.block1(x)
            features = cnn.block2(features)
            features = cnn.mask_early(features)
            features = cnn.block3(features)
            features = cnn.block4(features)
            features = cnn.block5(features)
            features = cnn.mask_late(features)
            features = torch.flatten(features, 1)
            cnn_features.append(features)
        cnn_features = torch.stack(cnn_features, dim=0)  # (n_cnns, batch, flattened_dim)

        # Average features
        vit_mean = vit_features.mean(dim=0)   # (batch, hidden_d)
        cnn_mean = cnn_features.mean(dim=0)   # (batch, flattened_dim)

        # Optional weighting
        if self.use_weighted:
            alpha = self.vit_weight
            vit_mean = alpha * vit_mean
            cnn_mean = (1 - alpha) * cnn_mean

        # Project features
        vit_proj = self.v_proj(vit_mean)
        cnn_proj = self.c_proj(cnn_mean)

        # Branch-wise L2 normalization (equalize magnitudes across modalities: images & HFE)
        vit_proj = F.normalize(vit_proj, p=2, dim=1, eps=1e-12)
        cnn_proj = F.normalize(cnn_proj, p=2, dim=1, eps=1e-12)

        # Handle HFE features
        fusion_features = [vit_proj, cnn_proj]
        if self.use_hfe:
            assert hfe_features is not None, "hfe_features must be provided when use_hfe is True"
            hfe_proj = self.hfe_proj(hfe_features)
            # Keep HFE on the same scale as the images
            hfe_proj = F.normalize(hfe_proj, p=2, dim=1, eps=1e-12)
            fusion_features.append(hfe_proj)

        # Fuse and classify
        fused = torch.cat(fusion_features, dim=1)
        logits = self.classifier(fused)

        if self.is_multi_label:
            return torch.sigmoid(logits)
        else:
            return logits
