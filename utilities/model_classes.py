"""
Miscellaneous classes and enumerations for supporting the training script.
"""

from enum import Enum

# --------------------------------------------------- Classes ---------------------------------------------------------
#
# region Classes

class ModelTypeEnum(Enum):
    VIT = "vit"
    CNN = "cnn"
    FUSION = "fusion"
    RESNET50 = "resnet50"
    INCEPTION_V3 = "inception_v3"
    ALEXNET = "alexnet"
    VIT_B_16 = "vit_b_16"
    VIT_B_32 = "vit_b_32"
    VIT_L_16 = "vit_l_16"
