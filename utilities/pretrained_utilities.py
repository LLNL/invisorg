"""
Miscellaneous functions for the pre-trained models used in the training script.
"""
import sys
import time
import os
import warnings
import csv
import glob
from datetime import timedelta
from enum import Enum

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.backends.mps
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import BCELoss, CrossEntropyLoss
from torchinfo import summary
from tabulate import tabulate
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision.models as models
from torchvision.models import ResNet50_Weights, Inception_V3_Weights, AlexNet_Weights
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights

import pandas as pd
import pickle
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss, jaccard_score, average_precision_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from tqdm import tqdm

sys.path.append(os.path.dirname(os.getcwd()))
from utilities.model_classes import ModelTypeEnum


np.random.seed(0)
torch.manual_seed(0)

sns.set_theme(style="darkgrid")
sns.set_context("paper")



# ----------------------------------------------- Pre-Trained Models --------------------------------------------------
#
# region Functions

class PretrainedModelWrapper(nn.Module):
    """
    Wrapper for pre-trained TorchVision models to handle different classification scenarios.

    This wrapper adapts pre-trained models for binary, multi-class, and multi-label classification
    by replacing the final classification layer and adding appropriate activation functions.
    """

    def __init__(self, model_type, num_classes, is_binary_class=False, is_multi_label=False,
                 dropout_rate=0.2, freeze_pretrained_backbone=False):
        """
        Initialize the pretrained model wrapper.

        Args:
            model_type (ModelTypeEnum): Type of model to load
            num_classes (int): Number of output classes
            is_binary_class (bool): Whether this is binary classification
            is_multi_label (bool): Whether this is multi-label classification
            freeze_pretrained_backbone (bool): Whether to freeze backbone parameters
            dropout_rate (float): Dropout rate for the classifier head
        """
        super(PretrainedModelWrapper, self).__init__()

        self.model_type = model_type
        self.num_classes = num_classes
        self.is_binary_class = is_binary_class
        self.is_multi_label = is_multi_label
        self._frozen = bool(freeze_pretrained_backbone)

        # Load the appropriate pretrained model
        if model_type == ModelTypeEnum.RESNET50:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier

        elif model_type == ModelTypeEnum.INCEPTION_V3:
            self.backbone = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
            # Also remove auxiliary classifier for inception
            self.backbone.aux_logits = False

        elif model_type == ModelTypeEnum.ALEXNET:
            self.backbone = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Identity()  # Remove original classifier

        elif model_type == ModelTypeEnum.VIT_B_16:
            self.backbone = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()  # Remove original classifier

        elif model_type == ModelTypeEnum.VIT_B_32:
            self.backbone = models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()  # Remove original classifier

        elif model_type == ModelTypeEnum.VIT_L_16:
            self.backbone = models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()  # Remove original classifier

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Linear Probe
        # Keep ImageNet weights: the model's backbone is frozen but the classifier head is trainable.
        for param in self.backbone.parameters():
            param.requires_grad = not self._frozen
        # keep BN/Dropout in eval mode if frozen (CNNs); ViTs use LayerNorm
        if self._frozen:
            self.backbone.eval()
            for m in self.backbone.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        # Create custom classifier head
        if is_binary_class:
            # Binary classification: single output with sigmoid
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 1)
            )
        elif is_multi_label:
            # Multi-label: multiple outputs with sigmoid
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes),
                nn.Sigmoid()
            )
        else:
            # Multi-class: multiple outputs with softmax (applied in loss function)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes)
            )


    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            # keep backbone frozen & in eval regardless of outer .train()
            self.backbone.eval()
            for m in self.backbone.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        return self


    def forward(self, x):
        """Forward pass through the model."""
        # Force correct input size for Inception-V3
        if self.model_type == ModelTypeEnum.INCEPTION_V3:
            # Inception-V3 expects 299x299, but 256x256 should work too
            if x.size(-1) != 299 or x.size(-2) != 299:
                x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Force correct input size for Vision Transformers
        if self.model_type in [ModelTypeEnum.VIT_B_16, ModelTypeEnum.VIT_B_32, ModelTypeEnum.VIT_L_16]:
            # Vision Transformers expect 224x224
            if x.size(-1) != 224 or x.size(-2) != 224:
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        features = self.backbone(x)
        logits   = self.classifier(features)  # shape: [B, 1] for binary, [B, C] otherwise

        # If it’s binary, turn that single logit into a two-class logit [–l, +l]
        if self.is_binary_class:
            # negative-class logit = –positive-class logit
            logits = torch.cat([-logits, logits], dim=1)  # now shape [B, 2]

        return logits
