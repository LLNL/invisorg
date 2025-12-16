
"""
Training script.
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
from tabulate import tabulate
from torchinfo import summary
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
from utilities.basic_utilities import get_print_time, create_dir, print_develop_message, restricted_float, test_size_float
from utilities.model_classes import ModelTypeEnum
from utilities.pretrained_utilities import PretrainedModelWrapper
from utilities.training_utilities import unwrap_subset, print_label_summary, compute_metrics, predict_with_model
from utilities.training_utilities import get_true_labels, model_type_enum, load_model, is_pretrained_model
from utilities.training_utilities import load_backbones_for_fusion, print_distribution_summary, compute_roc_auc_safe

from models.vit import MmViT
from models.cnn import MmSparseAwareCNN
from models.fusion import MmFusionModel
from datasets.dataset import PublicMicrobiomeDataset, ClassLabel

np.random.seed(0)
torch.manual_seed(0)

sns.set_theme(style="darkgrid")
sns.set_context("paper")


# ------------------------------------------------------- MAIN --------------------------------------------------------
#
# region Main
def main(args):
    """
    Main function call.
    """

    start_time_overall = time.time()

    # ----------------------------------------------- CLI Arguments ---------------------------------------------------
    #
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, type=str, metavar="NAME", help="Name of the dataset.")

    parser.add_argument("--images_path", required=True, type=str, metavar="PATH",
                        help="Path to directory with microbiome maps.")

    parser.add_argument("--output_dir", required=True, type=str, metavar="PATH",
                        help="Path to output directory.")

    parser.add_argument("--labels_file", required=True, type=str, help="File with labels.", metavar="PATH")

    parser.add_argument("--batch_size", required=True, type=int, help="Batch Size.", metavar="SIZE")

    parser.add_argument("--number_epochs", required=True, type=int, metavar="EPOCHS",
                        help="Number of Training Epochs.")

    # Mutually exclusive among "vit", "cnn", and "fusion".
    parser.add_argument("--model_type", required=True, type=model_type_enum,
                        help=f"Type of Model to Train. Options: {', '.join([e.value for e in ModelTypeEnum])}.")

    parser.add_argument("--dropout_rate", type=float, default=0.2, metavar="DROPOUT",
                        help="Dropout rate for classifier head. (default: %(default)s)")

    # Pretrained Models
    parser.add_argument("--freeze_pretrained_backbone", action="store_true",
                        help="Freeze pretrained backbone and train only the new head (linear probe).")

    # CNN-specific arguments: not required unless model_type is 'cnn'
    parser.add_argument("--cnn_hidden_d", type=int, default=None, metavar="HIDDEN_D",
                        help=f"(Required for '{ModelTypeEnum.CNN.value}') Size of the Hidden Dimension to map to.")

    # Vit-specific arguments: not required unless model_type is 'vit'
    parser.add_argument("--vit_hidden_d", type=int, default=None, metavar="HIDDEN_D",
                        help=f"(Required for '{ModelTypeEnum.VIT.value}') Size of the Hidden Dimension to map to.")
    parser.add_argument("--vit_number_patches", type=int, default=None, metavar="PATCHES",
                        help=f"(Required for '{ModelTypeEnum.VIT.value}') Number of Patches to split into.")
    parser.add_argument("--vit_number_blocks", type=int, default=None, metavar="BLOCKS",
                        help=f"(Required for '{ModelTypeEnum.VIT.value}') Number of Blocks.")
    parser.add_argument("--vit_number_heads", type=int, default=None, metavar="N_HEADS",
                        help=f"(Required for '{ModelTypeEnum.VIT.value}') Number of heads in each Transformer Encoder block.")

    # Fusion-specific arguments: Required if model_type is 'fusion'
    parser.add_argument("--fusion_vit_models_dir", required=False, type=str, metavar="VIT_MODELS_DIR",
                        help="Path to directory containing ViT backbone models (e.g., run_1/model_fold_1.pth etc.).")
    parser.add_argument("--fusion_cnn_models_dir", required=False, type=str, metavar="CNN_MODELS_DIR",
                        help="Path to directory containing CNN backbone models (e.g., run_1/model_fold_1.pth etc.).")
    parser.add_argument("--fusion_output_dim", required=False, type=int, metavar="FUSION_OUT_DIM", default=512,
                        help="Output Dimension of Fusion Model. (default: %(default)s)")
    parser.add_argument("--freeze_backbones", action="store_true", required=False,
                        help="If set, freeze all ViT and CNN backbone parameters during fusion model training.")

    # HFE features (fusion model only)
    parser.add_argument("--use_hfe", action="store_true", required=False,
                        help="Use HFE features for fusion model training.")
    parser.add_argument("--hfe_features", type=str, required=False, metavar="PATH",
                        help="Path to NPZ file containing HFE embeddings. Required when --use_hfe is True.")
    parser.add_argument("--hfe_dim", type=int, default=None, metavar="HFE_DIM",
                        help=f"(Required for '{ModelTypeEnum.FUSION.value}') Size of HFE embeddings.")

    # Other arguments
    parser.add_argument("--height_width", required=True, type=int, metavar="DIMENSIONS", default=256,
                        help="Image attributes for W x H. (default: %(default)s)")

    parser.add_argument("--n_channels", required=True, type=int, metavar="N_CHANNELS", default=3,
                        help="Number of Channels in the Images. (rgb default: %(default)s)")

    parser.add_argument("--learning_rate", required=True, type=restricted_float, metavar="RATE", default=0.0001,
                        help="Learning rate used for training (default: %(default)s)")

    parser.add_argument("--test_size", required=True, type=test_size_float, metavar="TEST_SIZE", default=0.1,
                        help="Size of the Holdout Test Set (default: %(default)s)")

    parser.add_argument("--n_runs", required=True, type=int, metavar="N_RUNS", default=1,
                        help="Number of Runs for Cross Validation. (default: %(default)s)")

    parser.add_argument("--k_folds", required=True, type=int, metavar="K_FOLDS", default=5,
                        help="Number of Folds for Cross Validation. (default: %(default)s)")

    parser.add_argument("--cores_per_socket", required=True, type=int, metavar="CORES_PER_SOCKET", default=64,
                        help="Number of CPU Cores per Socket. (default: %(default)s)")

    parser.add_argument("--target", required=True, type=str, metavar="TARGET_FACTOR",
                        help="Name of the Target Factor (class label) to train against.")

    parser.add_argument("--disease_only", action="store_true", required=False,
                        help="Use samples with disease status 'diseased' when target is DISEASE_TYPE or MULTI_LABEL.")

    parser.add_argument("--should_augment", action="store_true", required=False,
                        help="Augment the dataset across the selected dimensions (factors).")

    parser.add_argument("--develop_mode", action="store_true", required=False,
                        help="Development & Testing Mode. Sets the Training/Validation epochs to only 1% of the data'.")

    parser.add_argument("--one_fold", action="store_true", required=False,
                        help="Run the full program, but use just one fold of data.")


    # --------------------------------------------- Validate CLI ------------------------------------------------------
    #
    # Check that something was passed, and if it wasn't, then exit.
    if len(sys.argv) == 1:
        print("\nERROR: No arguments.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(args)

    # Validate required arguments based on model_type
    # Pretrained models don't need additional validation since they're self-contained.
    missing = []

    if args.model_type == ModelTypeEnum.CNN:
        if args.cnn_hidden_d is None:
            missing.append("--cnn_hidden_d")

    elif args.model_type == ModelTypeEnum.VIT:
        if args.vit_hidden_d is None:
            missing.append("--vit_hidden_d")
        if args.vit_number_patches is None:
            missing.append("--vit_number_patches")
        if args.vit_number_blocks is None:
            missing.append("--vit_number_blocks")
        if args.vit_number_heads is None:
            missing.append("--vit_number_heads")

    elif args.model_type == ModelTypeEnum.FUSION:
        # Fusion components
        if args.fusion_vit_models_dir is None:
            missing.append("--fusion_vit_models_dir")
        if args.fusion_cnn_models_dir is None:
            missing.append("--fusion_cnn_models_dir")
        if args.fusion_output_dim is None:
            missing.append("--fusion_output_dim")
        # ViT and CNN params are still needed to instantiate models
        if args.vit_hidden_d is None:
            missing.append("--vit_hidden_d")
        if args.vit_number_patches is None:
            missing.append("--vit_number_patches")
        if args.vit_number_blocks is None:
            missing.append("--vit_number_blocks")
        if args.vit_number_heads is None:
            missing.append("--vit_number_heads")
        if args.cnn_hidden_d is None:
            missing.append("--cnn_hidden_d")
        # HFE features
        if args.use_hfe and args.hfe_features is None:
            missing.append("--hfe_features (required when --use_hfe is True)")
        if args.use_hfe and args.hfe_dim is None:
            missing.append("--hfe_dim (required when --use_hfe is True)")

    if missing:
        parser.error(f"When --model_type is '{args.model_type.value}', the following arguments are required: " +
                     ", ".join(missing))

    # -----------------------------------------------------------------------------------------------------------------
    #
    #
    model_type            = args.model_type
    labels_file           = args.labels_file
    dataset_name          = args.dataset_name
    images_path           = args.images_path
    batch_size            = args.batch_size
    number_epochs         = args.number_epochs
    height_width          = args.height_width
    n_channels            = args.n_channels
    learning_rate         = args.learning_rate
    test_size             = args.test_size
    cnn_hidden_d          = args.cnn_hidden_d
    vit_hidden_d          = args.vit_hidden_d
    vit_number_patches    = args.vit_number_patches
    vit_number_blocks     = args.vit_number_blocks
    vit_number_heads      = args.vit_number_heads
    fusion_vit_models_dir = args.fusion_vit_models_dir
    fusion_cnn_models_dir = args.fusion_cnn_models_dir
    fusion_output_dim     = args.fusion_output_dim
    freeze_backbones      = args.freeze_backbones
    use_hfe               = args.use_hfe
    hfe_features          = args.hfe_features
    hfe_dim               = args.hfe_dim
    output_dir            = args.output_dir
    target_factor_args    = args.target
    disease_only          = args.disease_only
    n_runs                = args.n_runs
    k_folds               = args.k_folds
    cores_per_socket      = args.cores_per_socket
    develop_mode          = args.develop_mode
    one_fold              = args.one_fold

    # Set other flags/parameters.
    retain_kfold_ids   = False   # Writes the sample IDs to disk.
    should_augment     = False
    max_train_batches  = 50      # Number of training batches to process per epoch in develop mode.
    max_val_batches    = 30      # Number of validation batches to process per epoch in develop mode.
    class_support      = 8       # Threshold for dropping classes that do not have this many examples.

    # Pretrained
    freeze_backbone = args.freeze_backbone if hasattr(args, 'freeze_backbone') else False
    dropout_rate    = args.dropout_rate if hasattr(args, 'dropout_rate') else 0.2

    backbone_status = None
    if args.model_type == ModelTypeEnum.FUSION:
        backbone_status = "Frozen" if args.freeze_backbones else "Trainable"

    model_out_prefix = model_type.value
    assert model_out_prefix is not None, '[ERROR] model_out_prefix is NONE.'

    # Prevent HFE usage with non-Fusion models
    if use_hfe and model_type != ModelTypeEnum.FUSION:
        print(f"[{get_print_time()}] ERROR: HFE features can only be used with Fusion models!")
        print(f"[{get_print_time()}] Current model type is: {model_type.value}")
        sys.exit(1)

    # -----------------------------------------------------------------------------------------------------------------
    #
    # region Output Directories
    #
    if is_pretrained_model(model_type):
        if disease_only and target_factor_args in (ClassLabel.DISEASE_TYPE.value,
                                                   ClassLabel.MULTI_LABEL.value):
            output_dir = f'{output_dir}/{dataset_name}_disease_only/pretrained/{model_out_prefix}'
        else:
            output_dir = f'{output_dir}/{dataset_name}/pretrained/{model_out_prefix}'
    else:
        if disease_only and target_factor_args in (ClassLabel.DISEASE_TYPE.value, ClassLabel.MULTI_LABEL.value):
            output_dir = f'{output_dir}/{dataset_name}_disease_only/{model_out_prefix}'
        else:
            output_dir = f'{output_dir}/{dataset_name}/{model_out_prefix}'
        if args.model_type == ModelTypeEnum.FUSION:
            output_dir = f'{output_dir}_{backbone_status.lower()}'
    create_dir(with_path=output_dir)

    output_dir_models = f'{output_dir}/trained_models'
    create_dir(with_path=output_dir_models)

    output_dir_best_models = f'{output_dir}/best_models'
    create_dir(with_path=output_dir_best_models)

    output_dir_figures = f'{output_dir}/figures'
    create_dir(with_path=output_dir_figures)

    output_dir_metrics = f'{output_dir}/metrics'
    create_dir(with_path=output_dir_metrics)

    output_dir_kfold_ids = f'{output_dir}/kfold_ids'
    create_dir(with_path=output_dir_kfold_ids)

    # -----------------------------------------------------------------------------------------------------------------
    #
    # region Check Class Label

    # Check that the requested target factor is a valid one.
    possible_labels = [label.value for label in ClassLabel]
    if target_factor_args not in possible_labels:
        print(f"[{get_print_time()}] ERROR: Target Factor {target_factor_args} Not Valid!")
        print(f"[{get_print_time()}] Accepted Values: {possible_labels}")
        sys.exit(1)

    is_multi_label = False

    target_factor = None
    if target_factor_args == ClassLabel.DISEASE_STATUS.value:  # Binary
        target_factor = ClassLabel.DISEASE_STATUS
    if target_factor_args == ClassLabel.DISEASE_TYPE.value:    # Multi-Class
        target_factor = ClassLabel.DISEASE_TYPE
    if target_factor_args == ClassLabel.BODY_SITE.value:       # Multi-Class
        target_factor = ClassLabel.BODY_SITE
    if target_factor_args == ClassLabel.COUNTRY.value:         # Multi-Class
        target_factor = ClassLabel.COUNTRY
    if target_factor_args == ClassLabel.MULTI_LABEL.value:     # Multi-Label
        target_factor = ClassLabel.MULTI_LABEL
        is_multi_label = True

    assert target_factor is not None, '[ERROR] target_factor is NONE.'

    # -----------------------------------------------------------------------------------------------------------------
    #
    # "MPS" is the "Metal Performance Shader" in macOS Metal (AMD or Apple silicon).
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device_str = f'{device}'

    if develop_mode:
        print_develop_message()

    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}] ▸ Starting Model Training...")
    print(f"[{get_print_time()}]   - Model Type: {model_type.value}")
    print(f"[{get_print_time()}]   - Backbone Parameters: {backbone_status}")
    print(f"[{get_print_time()}]   - HFE: {use_hfe}")
    if use_hfe:
        print(f"[{get_print_time()}]   - HFE DIM: {hfe_dim}")
        print(f"[{get_print_time()}]   - HFE Path: {hfe_features}")
    print(f"[{get_print_time()}]   - Output Prefix: {model_out_prefix}")
    print(f"[{get_print_time()}]   - Disease Only Samples: {disease_only}")
    print(f"[{get_print_time()}]   - PyTorch Version: {torch.__version__}")
    print(f"[{get_print_time()}]   - CUDA Available: {torch.cuda.is_available()}")
    print(f"[{get_print_time()}]   - Device Count: {torch.cuda.device_count()}")
    print(f"[{get_print_time()}]   - Device: {device_str}")
    print(f"[{get_print_time()}]   - Number: {torch.cuda.device_count()}")
    print(f"[{get_print_time()}]")

    # -----------------------------------------------------------------------------------------------------------------
    #
    print(f"[{get_print_time()}] ▸ Dataset")
    print(f"[{get_print_time()}]   - Name Prefix: {dataset_name}")
    print(f"[{get_print_time()}]   - Target Factor: {target_factor.value}")
    print(f"[{get_print_time()}]   - Class Support: {class_support:,}")

    # ----------------------------------------------- Load Images -----------------------------------------------------
    #
    # region Dataset

    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}] ▸ Loading image resources...")
    print(f"[{get_print_time()}]   - Height x Width: {height_width}")
    print(f"[{get_print_time()}]")

    master_dataset = PublicMicrobiomeDataset(root_dir=images_path,
                                             csv_file=labels_file,
                                             class_support=class_support,
                                             should_augment=should_augment,
                                             height_width=height_width,
                                             target_factor=target_factor,
                                             disease_only=disease_only,
                                             use_hfe=use_hfe,
                                             hfe_features=hfe_features)

    classes_total = 0
    num_total_samples = len(master_dataset)
    if is_multi_label:
        # total number of channels = sum of uniques for each dim
        classes_total = sum(
            len(getattr(master_dataset, ClassLabel(col).mapping[0]))
            for col in master_dataset.dims
        )
    else:
        # single‑label: just the length of the one field
        classes_total = len(getattr(master_dataset, target_factor.mapping[0]))

    assert classes_total != 0, '[ERROR] classes_total is ZERO.'

    print(f"[{get_print_time()}] ▸ Total Samples: {num_total_samples:,}")
    print(f"[{get_print_time()}]   - All Classes: {classes_total:,}")
    print(f"[{get_print_time()}]")

    # ----------------------------------------- Train, Validation, Test -----------------------------------------------
    #
    #   The gimmick here will be to run the 'train_test_split()' twice on the Train dataset so that we get the
    #   Train, Validation, and Test datasets that we are after.
    #
    # region Train/Test Split

    print(f"[{get_print_time()}] ▸ Creating Train and Test datasets...")

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    dl_nworkers = cores_per_socket

    #   Initial Train and Test datasets.
    print(f"[{get_print_time()}]   - Creating Train/Test split...")

    # Ensure consistent DataFrame ordering for reproducible splits
    master_dataset.df_abundances = master_dataset.df_abundances.sort_values('sample_name').reset_index(drop=True)

    if is_multi_label:
        split_targets = np.stack(
            [master_dataset[i]['target'].numpy() for i in range(len(master_dataset))],axis=0)
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_seed)
    else:
        # 1D class labels for binary or multi‑class
        label_enum_column = master_dataset.target_factor.mapping[1]
        split_targets = master_dataset.df_abundances[label_enum_column].values
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_seed)

    train_indices, test_indices = next(
        splitter.split(
            X=np.zeros(len(master_dataset)),  # dummy features for getting indices.
            y=split_targets
        )
    )

    print(f"[{get_print_time()}]   - Creating Subsets...")
    ds_train = Subset(master_dataset, train_indices)
    ds_test  = Subset(master_dataset, test_indices)

    #   Data Loaders
    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}] ▸ Creating Dataloaders...")

    train_dataloader = DataLoader(ds_train, batch_size=batch_size, num_workers=dl_nworkers, shuffle=True,
                                  pin_memory=True, prefetch_factor=2)
    test_dataloader  = DataLoader(ds_test, batch_size=batch_size, num_workers=dl_nworkers, shuffle=False,
                                  pin_memory=True, prefetch_factor=2)

    overall_train_samples = len(train_dataloader.dataset)
    overall_test_samples  = len(test_dataloader.dataset)

    print(f"[{get_print_time()}]   - Training Samples: {overall_train_samples:,}")
    print(f"[{get_print_time()}]   - Testing Samples: {overall_test_samples:,}")

    if is_multi_label:
        print_distribution_summary("Full Dataset", master_dataset)
        print_distribution_summary("Training", ds_train)
        print_distribution_summary("Test", ds_test)

    print(f"[{get_print_time()}]")

    # ------------------------------------------ K-Fold Cross Validation -----------------------------------------------
    #
    # region Cross Validation

    print(f"[{get_print_time()}] ▸ Stratified K-Fold Cross Validation")
    print(f"[{get_print_time()}]   - Runs: {n_runs}")
    print(f"[{get_print_time()}]   - Folds: {k_folds}")
    print(f"[{get_print_time()}]   - Epochs: {number_epochs}")
    print(f"[{get_print_time()}]")

    # Global best model tracking across all runs and folds
    global_best_val_f1 = 0.0        # Track the single best validation F1 score
    global_best_model_state = None  # Store the single best model state
    global_best_run = 0             # Track which run had the best model
    global_best_fold = 0            # Track which fold had the best model

    global_metrics_history = {  # Global metrics aggregator for all runs and folds.
        'train': {
            'per_label_accuracy': [],
            'per_image_accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'precision_samples': [],
            'recall_samples': [],
            'f1_samples': [],
            'loss': [],
            'auc_roc': [],
            'auc_pr': [],
            'run': []
        },
        'val': {
            'per_label_accuracy': [],
            'per_image_accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'precision_samples': [],
            'recall_samples': [],
            'f1_samples': [],
            'loss': [],
            'auc_roc': [],
            'auc_pr': [],
            'run': []
        }
    }

    epoch_metrics_data = []

    for run in range(n_runs):
        print(f"[{get_print_time()}] ---------------------------------------------------------------------------")
        print(f"[{get_print_time()}] ❯ Run {run + 1} of {n_runs}")

        run_start_time = time.time()

        run_train_acc = []
        run_val_acc = []

        cross_val_ids = []  # Holds dataframes with each fold's ids for training/validation.

        cv_targets = None
        cv_splitter = None
        if is_multi_label:
            cv_targets = np.stack([ds_train[i]['target'].numpy() for i in range(len(ds_train))], axis=0)
            cv_splitter = MultilabelStratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed + run)
        else:
            base_dataset, indices = unwrap_subset(ds_train)
            label_enum_column = base_dataset.target_factor.mapping[1]
            class_indices = base_dataset.df_abundances.loc[indices, label_enum_column].values
            cv_targets = class_indices
            cv_splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed + run)
        assert cv_targets is not None, '[ERROR] cv_targets is NONE.'
        assert cv_splitter is not None, '[ERROR] cv_splitter is NONE.'

        # -------------------------------------------------------------------------------------------------------------
        #
        # Cross Validation Loop
        #
        for fold, (train_ids, val_ids) in enumerate(cv_splitter.split(X=np.zeros(len(ds_train)), y=cv_targets)):
            print(f"[{get_print_time()}] ---------------------------------------------------------------------------")
            print(f"[{get_print_time()}] ▸ Run {run + 1}: Fold {fold + 1} of {k_folds}")

            if develop_mode:
                print_develop_message()

            # Use ds_train as the base for subsampling
            train_subsampler = Subset(ds_train, train_ids)
            val_subsampler   = Subset(ds_train, val_ids)

            prefetch_factor = 64

            kf_train_dataloader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True,
                                             num_workers=dl_nworkers, pin_memory=True, prefetch_factor=prefetch_factor,
                                             persistent_workers=True)
            kf_validation_dataloader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False,
                                                  num_workers=dl_nworkers, pin_memory=True, prefetch_factor=prefetch_factor,
                                                  persistent_workers=True)

            # Compute and print dataset stats
            num_train_batches = len(kf_train_dataloader)
            num_val_batches   = len(kf_validation_dataloader)
            num_train_samples = len(kf_train_dataloader.dataset)
            num_val_samples   = len(kf_validation_dataloader.dataset)

            print(f"[{get_print_time()}]   - Training: {num_train_samples:,} samples in {num_train_batches:,} batches")
            print(f"[{get_print_time()}]   - Validation: {num_val_samples:,} samples in {num_val_batches:,} batches")

            if not develop_mode:
                if retain_kfold_ids:
                    print(f"[{get_print_time()}]   - Collecting sample IDs for training/validation sets...")

                    training_sample_ids = [sample['id'] for sample in kf_train_dataloader]
                    validation_sample_ids = [sample['id'] for sample in kf_validation_dataloader]

                    fold_ids = []

                    df_training = pd.DataFrame({'Sample_ID': training_sample_ids})
                    df_training['fold'] = fold
                    df_training['is_training'] = 1
                    df_training['is_validation'] = 0

                    df_validation = pd.DataFrame({'Sample_ID': validation_sample_ids})
                    df_validation['fold'] = fold
                    df_validation['is_training'] = 0
                    df_validation['is_validation'] = 1

                    fold_ids = pd.concat([df_training, df_validation], axis=0, ignore_index=True)
                    cross_val_ids.append(fold_ids)

            # Print a summary of the labels in the training and validation splits
            print(f"[{get_print_time()}]   - Creating Label Summary...")
            print_label_summary(train_subsampler, val_subsampler)

            # ---------------------------------------------------------------------------------------------------------
            #
            # region Model Setup
            #
            print(f"[{get_print_time()}] ▸ Preparing model...")

            is_binary_class = True
            loss_function = None
            if is_multi_label:  # Multi-Label
                is_binary_class = False
                loss_function = BCELoss()
            elif target_factor is ClassLabel.DISEASE_STATUS:  # Binary
                loss_function = BCELoss()
            else:
                loss_function = CrossEntropyLoss()
                is_binary_class = False
            assert loss_function is not None, '[ERROR] loss_function is NONE.'

            # Instantiate the model based on the selected model type.
            model = None

            if is_pretrained_model(model_type):
                print(f"[{get_print_time()}]   - Using pretrained model: {model_type.value}")
                print(f"[{get_print_time()}]   - Freeze backbone: {freeze_backbone}")

                model = PretrainedModelWrapper(
                    model_type=model_type,
                    num_classes=classes_total,
                    is_binary_class=is_binary_class,
                    is_multi_label=is_multi_label,
                    dropout_rate=dropout_rate,
                    freeze_pretrained_backbone=args.freeze_pretrained_backbone
                )

                # Always unwrap DataParallel before accessing .backbone / .classifier
                model_to_check = model.module if isinstance(model, nn.DataParallel) else model
                print(f"[{get_print_time()}]   - Backbone trainable?",
                      any(p.requires_grad for p in model_to_check.backbone.parameters()))
                print(f"[{get_print_time()}]   - Head trainable?",
                      any(p.requires_grad for p in model_to_check.classifier.parameters()))

            elif model_type == ModelTypeEnum.VIT:
                model = MmViT(chw=(n_channels, height_width, height_width),  # Input image shape (Channels, Height, Width).
                              n_patches=vit_number_patches,                  # Number of patches to split image into.
                              n_blocks=vit_number_blocks,                    # Number of Transformer Encoder blocks.
                              hidden_d=vit_hidden_d,                         # Size of hidden dimension to map to.
                              n_heads=vit_number_heads,                      # Heads in each Transformer Encoder block.
                              num_classes=classes_total,                       # Classification classes for Linear layer.
                              is_binary_class=is_binary_class,               # Are we doing binary classification.
                              is_multi_label=is_multi_label                  # Multi-label classification?
                              )

            elif model_type == ModelTypeEnum.CNN:
                model = MmSparseAwareCNN(num_classes=classes_total,
                                         input_size=(n_channels, height_width, height_width),
                                         hidden_d=cnn_hidden_d,
                                         is_multi_label=is_multi_label)

            elif model_type == ModelTypeEnum.FUSION:
                vit_models, cnn_models = load_backbones_for_fusion(
                    vit_models_dir=fusion_vit_models_dir,
                    cnn_models_dir=fusion_cnn_models_dir,
                    device=device,
                    is_multi_label=is_multi_label,
                    n_channels=n_channels,
                    height_width=height_width,
                    vit_number_patches=vit_number_patches,
                    vit_number_blocks=vit_number_blocks,
                    vit_hidden_d=vit_hidden_d,
                    vit_number_heads=vit_number_heads,
                    cnn_hidden_d=cnn_hidden_d,
                    num_classes=classes_total,
                    is_binary_class=is_binary_class,
                    freeze_backbones=freeze_backbones
                )

                model = MmFusionModel(
                    vit_models=vit_models,
                    cnn_models=cnn_models,
                    num_classes=classes_total,
                    output_dim=fusion_output_dim,
                    dropout_rate=0.2,
                    use_weighted=False,  # Optional: balance ViTs more
                    use_hfe=use_hfe,
                    hfe_dim=hfe_dim,
                    is_multi_label=is_multi_label
                )

            assert model is not None, '[ERROR] model is NONE.'

            # Optimizer setup
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable parameters (did you freeze everything?)")
            optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)

            # Scheduler setup
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

            # If more than one GPU is available, wrap the model for use on many GPUs. BUT, we have to ensure
            # that the model is only wrapped once, no matter how many folds we go through.
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
                    model = nn.DataParallel(model)
                    print(f"[{get_print_time()}]   - GPUs: {model.device_ids}")
            model.to(device)

            # Print to verify we're on the GPU. If GPU > 1, then reported device is where the main model copy is.
            print(f"[{get_print_time()}]   - Model Parameter Device: {next(model.parameters()).device}")

            if develop_mode:
                print(f"[{get_print_time()}]")
                print_develop_message()
            print(f"[{get_print_time()}]")

            history = {
                "train_acc": [],
                "train_loss": [],
                "val_acc": [],
                "val_loss": [],
                "test_acc": [],
                "test_loss": []
            }

            # Print a Keras-like summary of the model network.
            # nn.DataParallel wraps the model and copies the parameters for each GPU, but PyTorch counts
            # the DataParallel wrapper as a distinct module. So watch out. Leading 1 is dummy for batch‑size.
            model_to_summarize = model.module if isinstance(model, nn.DataParallel) else model

            # we need two dummy inputs: the image tensor and the HFE vector
            if getattr(model_to_summarize, "use_hfe", False):
                input_shapes = [
                    (1, n_channels, height_width, height_width),  # image
                    (1, hfe_dim),                                 # HFE features
                ]
                summary(model=model_to_summarize, input_size=input_shapes)
            else:
                summary(model=model_to_summarize, input_size=(1, n_channels, height_width, height_width))

            print(f"[{get_print_time()}]")
            print(f"[{get_print_time()}] ▸ Model Summary")
            print(f"[{get_print_time()}]   - Type: {model_out_prefix}")

            if is_pretrained_model(model_type):
                print(f"[{get_print_time()}]   - Backbone Frozen: {freeze_backbone}")
                print(f"[{get_print_time()}]   - Dropout Rate: {dropout_rate}")

            if model_type == ModelTypeEnum.CNN:
                print(f"[{get_print_time()}]   - CNN Hidden Dim: {cnn_hidden_d}")

            if model_type == ModelTypeEnum.VIT:
                print(f"[{get_print_time()}]   - ViT No. Patches: {vit_number_patches}")
                print(f"[{get_print_time()}]   - ViT No. Blocks: {vit_number_blocks}")
                print(f"[{get_print_time()}]   - ViT No. Heads: {vit_number_heads}")
                print(f"[{get_print_time()}]   - ViT Hidden Dim: {vit_hidden_d}")

            # ---------------------------------------------------------------------------------------------------------
            #
            # region Model Training
            #
            training_steps = len(kf_train_dataloader) // batch_size
            validation_steps = len(kf_validation_dataloader) // batch_size

            print(f"[{get_print_time()}]")
            print(f"[{get_print_time()}] ▸ Training Particulars:")
            print(f"[{get_print_time()}]   - No. Total Classes: {classes_total:,}")
            print(f"[{get_print_time()}]   - Class Support: {class_support:,}")
            print(f"[{get_print_time()}]   - Epochs: {number_epochs}")
            print(f"[{get_print_time()}]   - Learning Rate: {learning_rate}")
            print(f"[{get_print_time()}]   - Batch Size: {batch_size}")
            print(f"[{get_print_time()}]   - Training Steps: {training_steps}")
            print(f"[{get_print_time()}]   - Validation Steps: {validation_steps}")
            print(f"[{get_print_time()}]   - Binary Classification: {is_binary_class}")
            print(f"[{get_print_time()}]   - Multi-Label Classification: {is_multi_label}")
            print(f"[{get_print_time()}]   - Loss Function: {loss_function}")
            print(f"[{get_print_time()}]")

            total_samples = len(kf_train_dataloader.dataset) + len(kf_validation_dataloader.dataset)

            training_losses = []
            validation_losses = []

            # Initialize metrics history container, which we'll use later for plots
            metrics_history = {
                'train': {
                    'per_label_accuracy': [],
                    'per_image_accuracy': [],
                    'precision_macro': [],
                    'recall_macro': [],
                    'f1_macro': [],
                    'precision_samples': [],
                    'recall_samples': [],
                    'f1_samples': [],
                    'auc_roc': [],
                    'auc_pr': []
                },
                'val': {
                    'per_label_accuracy': [],
                    'per_image_accuracy': [],
                    'precision_macro': [],
                    'recall_macro': [],
                    'f1_macro': [],
                    'precision_samples': [],
                    'recall_samples': [],
                    'f1_samples': [],
                    'auc_roc': [],
                    'auc_pr': []
                }
            }

            print(f"[{get_print_time()}] ▸ Training Model...")
            print(f"[{get_print_time()}]")

            # Label smoothing.
            # The epsilon factor, ε, controls how much to soften binary targets.
            ε = 0.1

            last_train_postfix = {}

            # We'll show a TQDM progress bar with per-sample updates, as these are more granular and show progress
            # in terms of the actual number of images processed.
            for epoch in range(1, number_epochs + 1):
                with tqdm(total=total_samples, unit="sample", bar_format='{l_bar}{bar}|{postfix}') as e_pbar:
                    e_pbar.set_description(f'Run {run + 1}, Fold {fold + 1}, Epoch [{epoch}/{number_epochs}]')

                    total_training_loss = 0
                    total_validation_loss = 0

                    correct_train = 0
                    correct_validation = 0

                    training_loss = 0
                    validation_loss = 0

                    num_training_samples_seen   = 0
                    num_validation_samples_seen = 0

                    # ----------------------------------------------------------------------------------
                    #
                    # region Training Loop
                    #
                    model.train()

                    # Initialize lists to store predictions and ground truth
                    train_y_true_list = []
                    train_y_pred_list = []

                    # Store continuous scores for AUC
                    train_auc_true_list = []
                    train_auc_score_list = []

                    val_auc_true_list = []
                    val_auc_score_list = []

                    for idx, batch in enumerate(kf_train_dataloader):
                        train_images = batch['image']      # Torch tensor of rgb values for each image in the batch.
                        train_img_class = batch['class']   # Integer enumerated codes for the batch's enumerated labels.
                        if is_multi_label:
                            train_img_class = batch['target']

                        # Extract HFE features if available -- Only in Fusion model
                        train_hfe_features = None
                        if use_hfe and 'hfe_features' in batch:
                            train_hfe_features = batch['hfe_features'].to(device)

                        (train_images, train_img_class) = (train_images.to(device), train_img_class.to(device))

                        # Perform a forward pass, get training predictions, and calculate the training loss.
                        # The form of the predictions ('_pred') will depend on the type of classification we're doing.
                        train_pred = None
                        train_loss = None

                        # Forward pass with Images and HFE features
                        model_to_check = model.module if isinstance(model, nn.DataParallel) else model
                        if getattr(model_to_check, "use_hfe", False):
                            train_raw_outputs = model(train_images, hfe_features=train_hfe_features)
                        else:
                            train_raw_outputs = model(train_images)

                        # Build scores for AUC + compute loss + build labels for "discrete" metrics
                        if is_multi_label:
                            # outputs are already probabilities (Sigmoid head)
                            train_scores = train_raw_outputs  # (B, C) in [0,1]
                            y = train_img_class.float()
                            y_smooth = y * (1 - ε) + (ε / y.size(1))
                            train_loss = loss_function(train_scores, y_smooth)
                            train_pred = train_scores

                            # AUC collectors
                            train_auc_true_list.append(train_img_class.detach().cpu().numpy())
                            train_auc_score_list.append(train_scores.detach().cpu().numpy())

                        elif target_factor is ClassLabel.DISEASE_STATUS:
                            # Binary: model returns 2 logits; take positive class prob
                            pos_probs = torch.sigmoid(train_raw_outputs[:, 1])  # (B,)
                            train_loss = loss_function(pos_probs, train_img_class.float())
                            train_pred = pos_probs

                            # AUC collectors
                            train_auc_true_list.append(train_img_class.detach().cpu().numpy())
                            train_auc_score_list.append(pos_probs.detach().cpu().numpy())

                        else:
                            # Multi-class: use logits for CE; probabilities for AUC
                            train_loss = loss_function(train_raw_outputs, train_img_class)
                            train_probs = torch.softmax(train_raw_outputs, dim=1)  # (B, C)
                            train_classes = train_raw_outputs.argmax(dim=1)  # (B,)
                            train_pred = train_classes

                            # AUC collectors
                            train_auc_true_list.append(train_img_class.detach().cpu().numpy())
                            train_auc_score_list.append(train_probs.detach().cpu().numpy())

                        assert train_pred is not None, '[ERROR] train_pred is NONE.'
                        assert train_loss is not None, '[ERROR] train_loss is NONE.'

                        # Zero out the gradients, perform the backpropagation step, and update the weights
                        optimizer.zero_grad()
                        train_loss.backward()

                        optimizer.step()
                        scheduler.step()

                        # Add the loss to the total training loss so far and calculate the number of correct predictions.
                        training_loss = train_loss.item()  # Store it for plot.
                        total_training_loss += train_loss.item()

                        # Only threshold if we have probabilities (multi-label or binary).
                        if is_multi_label or target_factor is ClassLabel.DISEASE_STATUS:
                            train_bool_pred = (train_pred > 0.5).long()
                        else:
                            # multi-class: train_pred is already integer class indices
                            train_bool_pred = train_pred.long()

                        # The number of correct predictions in 'this' batch
                        num_correct_preds = None
                        if is_multi_label:
                            num_correct_preds = (train_bool_pred == train_img_class).sum().item()
                        elif target_factor is ClassLabel.DISEASE_STATUS:
                            num_correct_preds = (train_bool_pred == train_img_class).sum().item()
                        else:  # multi‑class
                            num_correct_preds = (train_pred == train_img_class).sum().item()

                        assert num_correct_preds is not None, '[ERROR] num_correct_preds is NONE.'

                        correct_train += num_correct_preds
                        num_training_samples_seen += train_bool_pred.size(0)

                        y_true = train_img_class.detach().cpu().numpy()
                        if y_true.ndim == 1:
                            y_true = y_true.reshape(-1, 1)

                        y_pred = train_bool_pred.detach().cpu().numpy()
                        if y_pred.ndim == 1:
                            y_pred = y_pred.reshape(-1, 1)

                        # Append these to lists for later metric computation
                        train_y_true_list.append(y_true)
                        train_y_pred_list.append(y_pred)

                        # Metrics to report in the progress bar.
                        train_metrics = compute_metrics(
                            y_true=np.vstack(train_y_true_list),
                            y_pred=np.vstack(train_y_pred_list),
                            is_multi_label=is_multi_label
                        )

                        potsfix_metrics = {
                            'T_Label': f"{train_metrics['per_label_accuracy']:.4f}",
                            'T_Image': f"{train_metrics['per_image_accuracy']:.4f}",
                            'T_Loss': f"{train_loss.item():.4f}",
                            "V_Label": "0.00",  # pre-filled for UI display.
                            "V_Image": "0.00",
                            "V_Loss": "0.00"
                        }

                        last_train_postfix = potsfix_metrics.copy()

                        # (optional) balanced‑acc for multi‑class
                        if not is_multi_label and target_factor in (
                                ClassLabel.DISEASE_TYPE,
                                ClassLabel.BODY_SITE,
                                ClassLabel.COUNTRY
                        ):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                balanced_acc = balanced_accuracy_score(y_true, y_pred)

                        # Update tqdm bar
                        e_pbar.set_postfix(potsfix_metrics)

                        # And also update the bar using the actual number of samples (images) in the batch.
                        current_batch_size = batch['image'].size(0)
                        e_pbar.update(current_batch_size)

                        # In develop mode, break after processing a few batches
                        if develop_mode and idx + 1 >= max_train_batches:
                            break

                    training_losses.append(training_loss)

                    # ----------------------------------------------------------------------------------
                    #
                    # region Validation Loop
                    #
                    with torch.no_grad():  # Switch off the gradient calculation
                        model.eval()
                        val_y_true_list = []
                        val_y_pred_list = []
                        for idx, batch in enumerate(kf_validation_dataloader):
                            val_images = batch['image']
                            val_img_class = batch['class']
                            if is_multi_label:
                                val_img_class = batch['target']

                            # Extract HFE features if available
                            val_hfe_features = None
                            if use_hfe and 'hfe_features' in batch:
                                val_hfe_features = batch['hfe_features'].to(device)

                            (val_images, val_img_class) = (val_images.to(device), val_img_class.to(device))

                            val_pred = None
                            val_loss = None

                            model_to_check = model.module if isinstance(model, nn.DataParallel) else model
                            if getattr(model_to_check, "use_hfe", False):
                                val_raw_outputs = model(val_images, hfe_features=val_hfe_features)
                            else:
                                val_raw_outputs = model(val_images)

                            if is_multi_label:
                                val_pred = val_raw_outputs  # shape: (B, C)
                                val_loss = loss_function(val_pred, val_img_class.float())
                                val_scores_for_auc = val_pred  # (B, C) probabilities

                            elif target_factor is ClassLabel.DISEASE_STATUS:
                                val_pred = torch.sigmoid(val_raw_outputs[:, 1])
                                val_loss = loss_function(val_pred, val_img_class.float())
                                val_scores_for_auc = val_pred  # (B,) probabilities

                            else:
                                # multi-class (type, site, country): raw logits + CrossEntropyLoss
                                val_loss = loss_function(val_raw_outputs, val_img_class)
                                val_pred = val_raw_outputs.argmax(dim=1)  # shape: (B,)
                                val_scores_for_auc = torch.softmax(val_raw_outputs, dim=1)  # (B, C) probabilities

                            assert val_pred is not None, '[ERROR] val_pred is NONE.'
                            assert val_loss is not None, '[ERROR] val_loss is NONE.'

                            # Continuous scores for AUC (do not alter val_pred semantics)
                            if is_multi_label:
                                val_scores_for_auc = val_pred  # (B, C) probabilities
                            elif target_factor is ClassLabel.DISEASE_STATUS:
                                val_scores_for_auc = val_pred  # (B,) probabilities
                            else:
                                val_scores_for_auc = torch.softmax(val_raw_outputs, dim=1)  # (B, C) probabilities

                            # AUC collectors
                            val_auc_true_list.append(val_img_class.detach().cpu().numpy())
                            val_auc_score_list.append(val_scores_for_auc.detach().cpu().numpy())

                            validation_loss = val_loss.item()
                            total_validation_loss += val_loss.item()

                            if is_multi_label or target_factor is ClassLabel.DISEASE_STATUS:
                                # only threshold when we have probabilities
                                val_bool_pred = (val_pred > 0.5).long()
                            else:
                                # multi-class: val_pred is already integer class indices
                                val_bool_pred = val_pred.long()

                            # Convert tensors to numpy arrays (detach from graph and move to CPU)
                            y_val_true = val_img_class.detach().cpu().numpy()
                            if y_val_true.ndim == 1:
                                y_val_true = y_val_true.reshape(-1, 1)

                            # For multi-label, predictions already thresholded:
                            y_val_pred = val_bool_pred.detach().cpu().numpy()
                            if y_val_pred.ndim == 1:
                                y_val_pred = y_val_pred.reshape(-1, 1)

                            # Append these to lists for later metric computation
                            val_y_true_list.append(y_val_true)
                            val_y_pred_list.append(y_val_pred)

                            if is_multi_label:
                                correct_validation += (val_bool_pred == val_img_class).sum().item()
                            elif target_factor is ClassLabel.DISEASE_STATUS:
                                correct_validation += (val_bool_pred == val_img_class).sum().item()
                            else:
                                correct_validation += (val_pred == val_img_class).sum().item()

                            num_validation_samples_seen += val_bool_pred.size(0)

                            val_metrics_live = compute_metrics(
                                y_true=np.vstack(val_y_true_list),
                                y_pred=np.vstack(val_y_pred_list),
                                is_multi_label=is_multi_label
                            )
                            val_postfix = {
                                "V_Label": f"{val_metrics_live['per_label_accuracy']:.4f}",
                                "V_Image": f"{val_metrics_live['per_image_accuracy']:.4f}",
                                "V_Loss": f"{val_loss.item():.4f}"
                            }
                            # Merge with the last training values for reporting in the TQDM bar.
                            merged_postfix = {**last_train_postfix, **val_postfix}
                            e_pbar.set_postfix(merged_postfix)

                            current_batch_size = batch['image'].size(0)
                            e_pbar.update(current_batch_size)

                            # In develop mode, break after processing a few batches
                            if develop_mode and idx + 1 >= max_val_batches:
                                break

                        validation_losses.append(validation_loss)

                    # ----------------------------------------------------------------------------------
                    #
                    # Accuracy Reports
                    #
                    train_denominator = num_training_samples_seen * (classes_total if is_multi_label else 1)
                    train_acc         = correct_train / train_denominator
                    train_loss        = total_training_loss / len(kf_train_dataloader)

                    val_denominator  = num_validation_samples_seen * (classes_total if is_multi_label else 1)
                    val_acc          = correct_validation / val_denominator
                    val_loss         = total_validation_loss / len(kf_validation_dataloader)

                    # Update History
                    history['train_acc'].append(train_acc)
                    history['train_loss'].append(train_loss)
                    history['val_acc'].append(val_acc)
                    history['val_loss'].append(val_loss)

                    # ------------------------------ Close the Update Bar ----------------------------------
                    #
                    e_pbar.update(total_samples - e_pbar.n)
                    e_pbar.close()

                    # --------------------------------- Epoch Reports --------------------------------------
                    #
                    # Concatenate predictions and ground truth arrays for training
                    train_y_true = np.concatenate(train_y_true_list, axis=0)
                    train_y_pred = np.concatenate(train_y_pred_list, axis=0)

                    val_y_true = np.concatenate(val_y_true_list, axis=0)
                    val_y_pred = np.concatenate(val_y_pred_list, axis=0)

                    # Compute Metrics
                    train_metrics = compute_metrics(y_true=train_y_true,
                                                    y_pred=train_y_pred,
                                                    is_multi_label=is_multi_label)

                    val_metrics = compute_metrics(y_true=val_y_true,
                                                  y_pred=val_y_pred,
                                                  is_multi_label=is_multi_label)

                    # ---- END-OF-EPOCH AUCs (uses continuous-score collectors) ----
                    # TRAIN
                    if is_multi_label:
                        ytr_true  = np.vstack(train_auc_true_list)  # (N, C)
                        ytr_score = np.vstack(train_auc_score_list)  # (N, C)
                    else:
                        ytr_true  = np.concatenate(train_auc_true_list).ravel()  # (N,)
                        if target_factor is ClassLabel.DISEASE_STATUS:  # binary: (N,)
                            ytr_score = np.concatenate(train_auc_score_list).ravel()
                        else:  # multiclass: (N, C)
                            ytr_score = np.vstack(train_auc_score_list)

                    train_auc_dict = compute_roc_auc_safe(ytr_true, ytr_score, is_multi_label, target_factor)

                    # VAL
                    if is_multi_label:
                        yv_true  = np.vstack(val_auc_true_list)  # (N, C)
                        yv_score = np.vstack(val_auc_score_list)  # (N, C)
                    else:
                        yv_true  = np.concatenate(val_auc_true_list).ravel()  # (N,)
                        if target_factor is ClassLabel.DISEASE_STATUS:  # binary: (N,)
                            yv_score = np.concatenate(val_auc_score_list).ravel()
                        else:  # multiclass: (N, C)
                            yv_score = np.vstack(val_auc_score_list)

                    val_auc_dict = compute_roc_auc_safe(yv_true, yv_score, is_multi_label, target_factor)

                    # Attach AUCs so the next block (append to metrics_history) works without KeyError
                    train_metrics['auc_roc'] = train_auc_dict['roc_auc_macro']
                    train_metrics['auc_pr']  = train_auc_dict['pr_auc_macro']
                    val_metrics['auc_roc']   = val_auc_dict['roc_auc_macro']
                    val_metrics['auc_pr']    = val_auc_dict['pr_auc_macro']

                    # Append metrics to history
                    for key in metrics_history['train']:
                        metrics_history['train'][key].append(train_metrics[key])
                        metrics_history['val'][key].append(val_metrics[key])

                    current_lr = optimizer.param_groups[0]['lr']
                    tqdm.write(f" ‣ Learning Rate: {current_lr:.6f}")

                    tqdm.write(" ‣ Training Metrics:")
                    tqdm.write(f"   - Per-label Accuracy: {train_metrics['per_label_accuracy'] * 100:.2f}%")
                    tqdm.write(f"   - Per-image Accuracy: {train_metrics['per_image_accuracy'] * 100:.2f}%")
                    tqdm.write(f"   - Precision (macro): {train_metrics['precision_macro']:.4f}")
                    tqdm.write(f"   - Recall (macro): {train_metrics['recall_macro']:.4f}")
                    tqdm.write(f"   - F1 (macro): {train_metrics['f1_macro']:.4f}")
                    tqdm.write(f"   - Precision (samples): {train_metrics['precision_samples']:.4f}")
                    tqdm.write(f"   - Recall (samples): {train_metrics['recall_samples']:.4f}")
                    tqdm.write(f"   - F1 (samples): {train_metrics['f1_samples']:.4f}")
                    tqdm.write(f"   - AUC ROC (macro): {train_metrics['auc_roc']:.4f}")
                    tqdm.write(f"   - AUC PR  (macro): {train_metrics['auc_pr']:.4f}")

                    tqdm.write(" ‣ Validation Metrics:")
                    tqdm.write(f"   - Per-label Accuracy: {val_metrics['per_label_accuracy'] * 100:.2f}%")
                    tqdm.write(f"   - Per-image Accuracy: {val_metrics['per_image_accuracy'] * 100:.2f}%")
                    tqdm.write(f"   - Precision (macro): {val_metrics['precision_macro']:.4f}")
                    tqdm.write(f"   - Recall (macro): {val_metrics['recall_macro']:.4f}")
                    tqdm.write(f"   - F1 (macro): {val_metrics['f1_macro']:.4f}")
                    tqdm.write(f"   - Precision (samples): {val_metrics['precision_samples']:.4f}")
                    tqdm.write(f"   - Recall (samples): {val_metrics['recall_samples']:.4f}")
                    tqdm.write(f"   - F1 (samples): {val_metrics['f1_samples']:.4f}")
                    tqdm.write(f"   - AUC ROC (macro): {val_metrics['auc_roc']:.4f}")
                    tqdm.write(f"   - AUC PR  (macro): {val_metrics['auc_pr']:.4f}")

                    # Check if this is the global best model
                    current_val_f1 = val_metrics['f1_macro']
                    if current_val_f1 > global_best_val_f1:
                        global_best_val_f1 = current_val_f1
                        global_best_model_state = model.state_dict().copy()  # Save global best model state
                        global_best_run = run + 1  # Track best run
                        global_best_fold = fold + 1  # Track best fold

                        tqdm.write(" ‣ New Best Model:")
                        tqdm.write(f"   - Run: {global_best_run}")
                        tqdm.write(f"   - Fold {global_best_fold}")
                        tqdm.write(f"   - Val F1: {global_best_val_f1:.4f}")

                    tqdm.write("")

                    # Record per-epoch metrics for Learning Curves Figure.
                    epoch_metrics_data.append({
                        'Run': run + 1,
                        'Fold': fold + 1,
                        'Epoch': epoch,
                        'Phase': 'Training',
                        'Loss': train_loss,
                        'per_label_accuracy': train_metrics['per_label_accuracy'] * 100,
                        'per_image_accuracy': train_metrics['per_image_accuracy'] * 100
                    })
                    epoch_metrics_data.append({
                        'Run': run + 1,
                        'Fold': fold + 1,
                        'Epoch': epoch,
                        'Phase': 'Validation',
                        'Loss': val_loss,
                        'per_label_accuracy': val_metrics['per_label_accuracy'] * 100,
                        'per_image_accuracy': val_metrics['per_image_accuracy'] * 100
                    })

            # ---------------------------------------------------------------------------------------------------------
            #
            # region Accuracy Reports
            #
            # Use a consistent denominator for computing accuracy.
            if is_multi_label:
                total_train_samples = num_training_samples_seen * classes_total
                total_val_samples   = num_validation_samples_seen * classes_total
            else:
                total_train_samples = num_training_samples_seen
                total_val_samples   = num_validation_samples_seen

            avg_training_acc = float(correct_train) / float(total_train_samples) * 100
            avg_val_acc  = float(correct_validation) / float(total_val_samples) * 100

            # (Optional) Update history if needed, so that it reflects the adjusted values:
            history['train_acc'][-1] = avg_training_acc / 100  # store as fraction
            history['val_acc'][-1]   = avg_val_acc / 100  # store as fraction

            print(f"[{get_print_time()}]")
            print(f"[{get_print_time()}] ▸ Training Results (Fold {fold + 1}):")
            print(f"[{get_print_time()}]   - Training Acc: {avg_training_acc:.2f}%")
            print(f"[{get_print_time()}]   - Training Loss: {history['train_loss'][-1]:.4f}")

            if is_multi_label:
                print(f"[{get_print_time()}]   - No. of Samples Seen: {total_train_samples} (multi-label)")
            else:
                print(f"[{get_print_time()}]   - No. of Samples Seen: {total_train_samples}")
            print(f"[{get_print_time()}]   - Got {correct_train} / {total_train_samples} ({avg_training_acc:.2f}% acc)")

            print(f"[{get_print_time()}]")
            print(f"[{get_print_time()}] ▸ Validation Results (Fold {fold + 1}):")
            print(f"[{get_print_time()}]   - Validation Acc: {avg_val_acc:.2f}%")
            print(f"[{get_print_time()}]   - Validation Loss: {history['val_loss'][-1]:.4f}")
            if is_multi_label:
                print(f"[{get_print_time()}]   - No. of Samples Seen: {total_val_samples} (multi-label)")
            else:
                print(f"[{get_print_time()}]   - No. of Samples Seen: {total_val_samples}")
            print(f"[{get_print_time()}]   - Got {correct_validation} / {total_val_samples} ({avg_val_acc:.2f}% acc)")

            run_train_acc.append(avg_training_acc)
            run_val_acc.append(avg_val_acc)

            # ---------------------------------------------------------------------------------------------------------
            #
            # At the end of each fold, after training for all epochs:
            #
            final_epoch_index = -1  # We use the last epoch's value
            for key in metrics_history['train']:
                final_train_value = metrics_history['train'][key][final_epoch_index]
                global_metrics_history['train'][key].append(final_train_value)

            for key in metrics_history['val']:
                final_val_value = metrics_history['val'][key][final_epoch_index]
                global_metrics_history['val'][key].append(final_val_value)

            # Append the final training loss for this fold:
            global_metrics_history['train']['loss'].append(history['train_loss'][final_epoch_index])
            global_metrics_history['train']['run'].append(run + 1)

            # Append the final validation loss for this fold:
            global_metrics_history['val']['loss'].append(history['val_loss'][final_epoch_index])
            global_metrics_history['val']['run'].append(run + 1)

            # ---------------------------------------------------------------------------------------------------------
            #
            # region Save Fold Model.
            model_output_dir_run = f"{output_dir_models}/run_{run+1}"
            model_save_path = f"{model_output_dir_run}/model_fold_{fold+1}.pth"
            create_dir(with_path=model_output_dir_run)
            torch.save(model.state_dict(), model_save_path)

            # ---------------------------------------------------------------------------------------------------------
            #
            # region End Fold Iteration
            #
            print(f"[{get_print_time()}]")

        # -------------------------------------------------------------------------------------------------------------
        #
        # Run End
        #
        run_end_time = time.time()
        run_runtime = run_end_time - run_start_time

        print(f"[{get_print_time()}] ---------------------------------------------------------------------------")
        print(f"[{get_print_time()}] ▶︎▶︎ Run {run + 1} END ◀︎◀︎")
        print(f"[{get_print_time()}]     - Runtime: {timedelta(seconds=run_runtime)}")

        if not develop_mode:
            if retain_kfold_ids:
                print(f"[{get_print_time()}]")
                print(f"[{get_print_time()}]  - Saving Training Sample IDs for this run...")
                cross_val_ids = pd.concat(cross_val_ids, axis=0)
                cross_val_ids_file = f'{output_dir_kfold_ids}/run_{run + 1}_of_{n_runs}.tab'
                cross_val_ids.to_csv(cross_val_ids_file, header=True, index=False, sep='\t', quoting=csv.QUOTE_NONE)

    # -------------------------------------------- Training Metrics ---------------------------------------------------
    #
    # region Training Metric Agg
    #
    print(f"[{get_print_time()}] ---------------------------------------------------------------------------")
    print(f"[{get_print_time()}] ▸ Aggregating Metrics for Run {run + 1}...")

    phase_mapping = {'train': 'Training', 'val': 'Validation'}
    aggregated_metrics = {'train': {}, 'val': {}}

    for phase in ['train', 'val']:
        phase_label = phase_mapping[phase]
        print(f"[{get_print_time()}]   - {phase_label} Phase")
        for metric in global_metrics_history[phase]:
            values = global_metrics_history[phase][metric]
            mean_value = np.mean(values)
            std_value = np.std(values)
            aggregated_metrics[phase][metric] = {'mean': mean_value, 'std': std_value, 'values': values}

            metric_label = metric.replace("_", " ").title()
            print(f"[{get_print_time()}]     - {metric_label}: Mean = {mean_value:.4f}, StD = {std_value:.4f}")

    mean_train_loss = np.mean(global_metrics_history['train']['loss'])
    std_train_loss  = np.std(global_metrics_history['train']['loss'])
    mean_val_loss   = np.mean(global_metrics_history['val']['loss'])
    std_val_loss    = np.std(global_metrics_history['val']['loss'])

    print(f"[{get_print_time()}]     - Training Loss: Mean = {mean_train_loss:.4f}, StD = {std_train_loss:.4f}")
    print(f"[{get_print_time()}]     - Validation Loss: Mean = {mean_val_loss:.4f}, StD = {std_val_loss:.4f}")

    # Save aggregated_metrics to a pickle file
    aggregated_metrics_file = os.path.join(output_dir_metrics, f"{model_out_prefix}_aggregated_metrics.pkl")
    with open(aggregated_metrics_file, "wb") as f:
        pickle.dump(aggregated_metrics, f)

    # Save global_metrics_history to a pickle file
    global_metrics_history_file = os.path.join(output_dir_metrics, f"{model_out_prefix}_global_metrics_history.pkl")
    with open(global_metrics_history_file, "wb") as f:
        pickle.dump(global_metrics_history, f)

    # Save epoch_metrics_data to a pickle file.
    epoch_metrics_data_file = os.path.join(output_dir_metrics, f"{model_out_prefix}_epoch_metrics_data.pkl")
    with open(epoch_metrics_data_file, "wb") as f:
        pickle.dump(epoch_metrics_data, f)

    # Save the single best model across all runs and folds
    if global_best_model_state is not None:
        if model_type == ModelTypeEnum.VIT:
            best_model_dir = f"{output_dir_best_models}/vit"  # ViT subdirectory
        elif model_type == ModelTypeEnum.CNN:
            best_model_dir = f"{output_dir_best_models}/cnn"  # CNN subdirectory
        else:
            best_model_dir = output_dir_best_models  # Fusion and others stay in root of output_dir_best_models

        create_dir(with_path=best_model_dir)  # Create subdirectory
        best_model_save_path = f"{best_model_dir}/{model_out_prefix}_best_model.pth"  # Save to subdirectory
        torch.save(global_best_model_state, best_model_save_path)

        print(f"[{get_print_time()}]")
        print(f"[{get_print_time()}] ▸ Saved Best Model:")
        print(f"[{get_print_time()}]   - Best Val F1: {global_best_val_f1:.4f}")
        print(f"[{get_print_time()}]   - From Run {global_best_run}, Fold {global_best_fold}")
        print(f"[{get_print_time()}]   - Saved to: {best_model_save_path}")
    else:
        print(f"[{get_print_time()}]   - WARNING: No best model state was saved!")

    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}] ---------------------------------------------------------------------------------")
    print(f"[{get_print_time()}] ------------------------------- Training Complete -------------------------------")
    print(f"[{get_print_time()}] ---------------------------------------------------------------------------------")
    print(f"[{get_print_time()}]")

    # ------------------------------------------- Testing & Ensembling ------------------------------------------------
    #
    # region Testing
    #
    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}] ▸ Testing Models...")

    aggregation_schemes = {
        "average": lambda preds: torch.mean(preds, dim=0),
        "max": lambda preds: torch.max(preds, dim=0)[0],
        "min": lambda preds: torch.min(preds, dim=0)[0]
    }

    # Choose an aggregation method (e.g., "average", "max", or "min")
    aggregation_method = "average"
    print(f"[{get_print_time()}]   - Ensembling Predictions: {aggregation_method.title()}")

    # List to collect predictions from all fold models.
    ensemble_predictions = []

    # Loop over Runs and Folds to load each model and predict on the Test set.
    for run in range(n_runs):
        model_output_dir_run = f"{output_dir_models}/run_{run+1}"
        for fold in range(k_folds):
            # Construct the model file path (assumes models were saved with this naming scheme)
            model_save_path = os.path.join(model_output_dir_run, f"model_fold_{fold+1}.pth")

            # Skip if the checkpoint doesn't exist
            if not os.path.exists(model_save_path):
                print(f"[{get_print_time()}] ⚠️ Model not found for fold {fold+1}: {model_save_path}, skipping.")
                continue

            # --------------------------------- Load Models for Testing ---------------------------------
            #
            if is_pretrained_model(model_type):
                model = PretrainedModelWrapper(
                    model_type=model_type,
                    num_classes=classes_total,
                    is_binary_class=is_binary_class,
                    is_multi_label=is_multi_label,
                    dropout_rate=dropout_rate,
                    freeze_pretrained_backbone=args.freeze_pretrained_backbone
                )
                # Load state dict and handle DataParallel prefix
                state_dict = torch.load(model_save_path, map_location=device)

                # Check if "module." prefix exists in keys (from DataParallel)
                has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

                if has_module_prefix:
                    # Remove "module." prefix from keys
                    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                else:
                    new_state_dict = state_dict

                model.load_state_dict(new_state_dict)
                model.to(device)
                model.eval()

            # Load the model with the load_model() function.
            elif model_type == ModelTypeEnum.VIT:
                model = load_model(MmViT, model_save_path, device, withDataParallel=True,
                                   chw=(n_channels, height_width, height_width),
                                   n_patches=vit_number_patches,
                                   n_blocks=vit_number_blocks,
                                   hidden_d=vit_hidden_d,
                                   n_heads=vit_number_heads,
                                   num_classes=classes_total,
                                   is_binary_class=is_binary_class,
                                   is_multi_label=is_multi_label)

            elif model_type == ModelTypeEnum.CNN:
                model = load_model(MmSparseAwareCNN, model_save_path, device, withDataParallel=True,
                                   num_classes=classes_total,
                                   input_size=(n_channels, height_width, height_width),
                                   hidden_d=cnn_hidden_d,
                                   is_multi_label=is_multi_label)

            elif model_type == ModelTypeEnum.FUSION:
                vit_models, cnn_models = load_backbones_for_fusion(
                    vit_models_dir=fusion_vit_models_dir,
                    cnn_models_dir=fusion_cnn_models_dir,
                    device=device,
                    is_multi_label=is_multi_label,
                    n_channels=n_channels,
                    height_width=height_width,
                    vit_number_patches=vit_number_patches,
                    vit_number_blocks=vit_number_blocks,
                    vit_hidden_d=vit_hidden_d,
                    vit_number_heads=vit_number_heads,
                    cnn_hidden_d=cnn_hidden_d,
                    num_classes=classes_total,
                    is_binary_class=is_binary_class,
                    freeze_backbones=freeze_backbones
                )

                model = load_model(MmFusionModel, model_save_path, device, withDataParallel=True,
                                   vit_models=vit_models,
                                   cnn_models=cnn_models,
                                   num_classes=classes_total,
                                   output_dim=fusion_output_dim,
                                   dropout_rate=0.2,
                                   use_weighted=False,
                                   use_hfe=use_hfe,
                                   hfe_dim=hfe_dim,
                                   is_multi_label=is_multi_label)

            # Get predictions on the test set using the predict_with_model() function.
            preds = predict_with_model(model, test_dataloader, device)
            ensemble_predictions.append(preds)

    # Stack predictions: shape will be [n_models, n_samples, n_outputs]
    ensemble_predictions_tensor = torch.stack(ensemble_predictions)

    # Aggregate predictions across models using the chosen aggregation scheme.
    aggregated_predictions = aggregation_schemes[aggregation_method](ensemble_predictions_tensor)

    # Convert the aggregated predictions tensor to a NumPy array. Used in ROC curves.
    aggregated_predictions_np = aggregated_predictions.cpu().detach().numpy()

    # For multi-label classification, threshold aggregated predictions to get binary outputs.
    final_predictions = None
    if is_multi_label:
        final_predictions = (aggregated_predictions >= 0.5).long()
    elif target_factor is ClassLabel.DISEASE_STATUS:
        final_predictions = (torch.sigmoid(aggregated_predictions[:, 1]) >= 0.5).long()
    else:
        final_predictions = aggregated_predictions.argmax(dim=1)
    assert final_predictions is not None, '[ERROR] final_predictions is NONE.'

    # Get true labels from the test set.
    true_labels = get_true_labels(test_dataloader, device, is_multi_label)

    # Collect necessary data for loss computation.
    preds_for_loss  = None
    labels_for_loss = None

    if is_multi_label:
        # Multi-label: already probabilities from sigmoid head
        preds_for_loss  = aggregated_predictions
        labels_for_loss = true_labels.float()
    elif target_factor is ClassLabel.DISEASE_STATUS:
        # Binary: aggregated_predictions are raw logits with shape [N, 2]
        # Select the positive class (index 1) and apply sigmoid
        probs_for_loss  = torch.sigmoid(aggregated_predictions[:, 1])  # shape: [N]
        preds_for_loss  = probs_for_loss
        labels_for_loss = true_labels.float()
    else:
        # Multi-class (DISEASE_TYPE, BODY_SITE, COUNTRY)
        preds_for_loss  = aggregated_predictions  # raw logits
        labels_for_loss = true_labels.long()

    assert preds_for_loss is not None and labels_for_loss is not None, \
        "[ERROR] preds_for_loss or labels_for_loss not set"

    # Compute Loss
    test_loss = loss_function(preds_for_loss, labels_for_loss).item()

    # Compute test accuracy as the elementwise mean (i.e. average over all label predictions).
    # This compares two shape-[N] tensors (for binary and multiclass), or two shape-[N,C] tensors (for multilabel)
    test_accuracy = (final_predictions == true_labels).float().mean().item()

    # Convert predictions and true labels to NumPy arrays for the classification report.
    final_predictions_np = final_predictions.numpy()
    true_labels_np = true_labels.numpy()

    # Classification Report
    #   Sklearn function returns a string. Set 'output_dict=True' to get a dictionary instead.
    #   'zero_division' sets precision/recall to 0 for labels with no predicted samples without a runtime warning.
    #   target_names = master_dataset.mln_class_names if is_multi_label else master_dataset.unique_labels
    target_names = (
        master_dataset.mln_class_names
        if is_multi_label
        else getattr(master_dataset, target_factor.mapping[0])
    )

    class_report = classification_report(y_true=true_labels_np,
                                         y_pred=final_predictions_np,
                                         target_names=target_names,  # Appearance order matters.
                                         zero_division=0,
                                         output_dict=False)

    class_report_dictionary = classification_report(y_true=true_labels_np,
                                                    y_pred=final_predictions_np,
                                                    target_names=target_names,
                                                    zero_division=0,
                                                    output_dict=True)

    # We want to display the results sorted by F1 Score (ascending), and this Df allows us to do that.
    df_f1_metrics = pd.DataFrame({
        "Class": target_names,
        "F1 Score": [class_report_dictionary[cls]['f1-score'] for cls in target_names],
        "Support": [class_report_dictionary[cls]['support'] for cls in target_names]
    })
    df_f1_metrics = df_f1_metrics.sort_values("F1 Score")

    # Bundle testing metrics into a dictionary.
    test_metrics = {
        "classification_report": class_report_dictionary,
        "df_f1_metrics": df_f1_metrics
    }

    # Compute Hamming Loss:
    h_loss = hamming_loss(true_labels_np, final_predictions_np)

    # Compute Jaccard Score (for multi-label, use 'samples' average)
    jaccard = None
    if is_multi_label:
        jaccard = jaccard_score(true_labels_np, final_predictions_np, average='samples')
    elif target_factor is ClassLabel.DISEASE_STATUS:
        # binary Jaccard (IoU) on the positive class
        jaccard = jaccard_score(true_labels_np, final_predictions_np, average='binary', zero_division=0)
    else:
        # multi-class: choose 'macro' as it averages per class
        jaccard = jaccard_score(true_labels_np, final_predictions_np, average='macro', zero_division=0)
    assert jaccard is not None, "[ERROR] jaccard was never set"

    # Compute precision, recall, and F1 for each class (micro and macro)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels_np,
                                                                                 final_predictions_np,
                                                                                 average='micro',
                                                                                 zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels_np,
                                                                                 final_predictions_np,
                                                                                 average='macro',
                                                                                 zero_division=0)

    # Mean Precision score (mAP) for each label, and then average.
    #   But before, we have to ensure 2D arrays: (n_samples, 1) for binary, (n_samples, C) otherwise
    if aggregated_predictions_np.ndim == 1:
        aggregated_predictions_np = aggregated_predictions_np.reshape(-1, 1)
    if true_labels_np.ndim == 1:
        true_labels_np = true_labels_np.reshape(-1, 1)

    mAPs = []
    if target_factor is ClassLabel.DISEASE_STATUS:  # Binary case
        y_true_bin  = true_labels_np.ravel()
        # Extract positive class probabilities for ROC curves
        y_score_bin = torch.sigmoid(torch.from_numpy(aggregated_predictions_np[:, 1])).numpy()
        mAPs = [average_precision_score(y_true_bin, y_score_bin)]

    elif target_factor in (  # Multi‑Class case
            ClassLabel.DISEASE_TYPE,
            ClassLabel.BODY_SITE,
            ClassLabel.COUNTRY):

        C = aggregated_predictions_np.shape[1]
        y = true_labels_np.ravel()
        y_onehot = label_binarize(y, classes=np.arange(C))  # Shape (N, C)
        for c in range(C):
            mAPs.append(average_precision_score(y_onehot[:, c], aggregated_predictions_np[:, c]))

    else:  # Multi‑Label case
        C = aggregated_predictions_np.shape[1]
        for c in range(C):
            mAPs.append(average_precision_score(true_labels_np[:, c], aggregated_predictions_np[:, c]))

    assert len(mAPs) > 0, "[ERROR] Mean Average Precision list 'mAPs' is empty."
    mAP = np.mean(mAPs)

    # ---- Test AUC ROC / AUC PRC ----
    if target_factor is ClassLabel.DISEASE_STATUS:
        # y_score_bin already computed above
        y_true_test  = true_labels_np.ravel()
        y_score_test = y_score_bin
    elif is_multi_label:
        y_true_test  = true_labels_np                                  # (N, C)
        y_score_test = aggregated_predictions_np                       # (N, C) probabilities
    else:
        # Multi-class: convert logits to probabilities for AUC
        z = aggregated_predictions_np
        z = z - np.max(z, axis=1, keepdims=True)
        y_score_test = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        y_true_test  = true_labels_np.ravel()

    test_auc_dict = compute_roc_auc_safe(y_true_test, y_score_test, is_multi_label, target_factor)
    test_auroc = test_auc_dict['roc_auc_macro']
    test_auprc = test_auc_dict['pr_auc_macro']

    # ------------------------------------------------------------------------------------------------
    #
    # Print the Reports to STDOUT
    #
    if is_multi_label:
        multi_test_metrics = compute_metrics(
            y_true=true_labels_np,
            y_pred=final_predictions_np,
            is_multi_label=True
        )
        per_label_test_acc = multi_test_metrics['per_label_accuracy']
        per_image_test_acc = multi_test_metrics['per_image_accuracy']
        print(f"[{get_print_time()}]   - Per-Label Test Accuracy: {per_label_test_acc * 100:.2f}%")
        print(f"[{get_print_time()}]   - Per-Image Test Accuracy: {per_image_test_acc * 100:.2f}%")

    print(f"[{get_print_time()}]   - Overall Test Accuracy: {test_accuracy* 100:.2f}")
    print(f"[{get_print_time()}]   - Overall Test Loss: {test_loss:.4f}")
    print(f"[{get_print_time()}]   - Hamming Loss: {h_loss:.4f}")
    print(f"[{get_print_time()}]   - Jaccard Score: {jaccard:.4f}")
    print(f"[{get_print_time()}]   - Micro Precision: {precision_micro:.4f}")
    print(f"[{get_print_time()}]   - Micro Recall: {recall_micro:.4f}")
    print(f"[{get_print_time()}]   - Micro F1: {f1_micro:.4f}")
    print(f"[{get_print_time()}]   - Macro Precision: {precision_macro:.4f}")
    print(f"[{get_print_time()}]   - Macro Recall: {recall_macro:.4f}")
    print(f"[{get_print_time()}]   - Macro F1: {f1_macro:.4f}")
    print(f"[{get_print_time()}]   - Mean Average Precision (mAP): {mAP:.4f}")
    print(f"[{get_print_time()}]   - AUROC (macro): {test_auroc:.4f}")
    print(f"[{get_print_time()}]   - AUPRC (macro): {test_auprc:.4f}")
    print(f"[{get_print_time()}]   - Classification Report")
    print(f"[{get_print_time()}]")
    print(class_report)

    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}] ▸ Saving testing metrics...")
    additional_metrics = {
        "hamming_loss": h_loss,
        "jaccard_score": jaccard,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "micro_f1": f1_micro,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "mean_average_precision": mAP,
        "auroc_macro": test_auroc,
        "auprc_macro": test_auprc
    }
    # Add the additional metrics to the test_metrics dictionary.
    test_metrics["additional_metrics"] = additional_metrics

    test_metrics_file = os.path.join(output_dir_metrics, f"{model_out_prefix}_test_metrics.pkl")
    with open(test_metrics_file, "wb") as f:
        pickle.dump(test_metrics, f)

    # Save the aggregated predictions as a NumPy array to a file.
    if target_factor is ClassLabel.DISEASE_STATUS:
        # For binary, save the positive class probabilities
        predictions_to_save = torch.sigmoid(torch.from_numpy(aggregated_predictions_np[:, 1])).numpy()
    else:
        predictions_to_save = aggregated_predictions_np
    aggregated_predictions_file = os.path.join(output_dir_metrics, f"{model_out_prefix}_aggregated_predictions.npy")
    np.save(aggregated_predictions_file, predictions_to_save)

    # Save 'true_labels_np' for ROC curves
    true_labels_file = os.path.join(output_dir_metrics, f"{model_out_prefix}_true_labels.npy")
    np.save(true_labels_file, true_labels_np)

    print(f"[{get_print_time()}]")

    # ------------------------------------------------- Figures -------------------------------------------------------
    #
    # region Figures
    #
    print(f"[{get_print_time()}] ▸ Creating figures...")
    print(f"[{get_print_time()}]")

    fig_dpi = 150

    fig_title_prefix = ""
    if target_factor is ClassLabel.MULTI_LABEL:
        fig_title_prefix = "Multi-Label Classification"
    else:
        # e.g. "BODY_SITE" → "Body Site", "DISEASE_STATUS" → "Disease Status"
        pretty_name      = target_factor.name.replace("_", " ").title()
        fig_title_prefix = f"{pretty_name} Classification"
    assert fig_title_prefix != "", "[ERROR] fig_title_prefix not set"

    # -----------------------------------------------------------------------------------
    #
    # Boxplot
    #
    # Create the boxplot_data list without including the "run" key.
    boxplot_data = []
    for phase in ['train', 'val']:
        for metric in global_metrics_history[phase]:
            # Skip the "run" key because it's not a metric.
            if metric in ("run", "loss"):
                continue
            # Format the metric name by replacing underscores with spaces and applying title-case.
            metric_label = metric.replace("_", " ").title()
            for value in global_metrics_history[phase][metric]:
                boxplot_data.append({
                    'Phase': phase_mapping[phase],
                    'Metric': metric_label,
                    'Value': value
                })

    df_box = pd.DataFrame(boxplot_data)

    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(x="Metric", y="Value", hue="Phase", data=df_box)
    plt.title(f"{fig_title_prefix} | {model_out_prefix.upper()} Metrics | {n_runs} Runs and {k_folds} Folds")
    plt.xticks(rotation=0)
    plt.xlabel("Metric", fontweight="bold")  # Bold x-axis label
    ax.legend(title="", loc='upper right')  # Remove the legend title

    # Add vertical dashed lines between metric categories.
    unique_metrics = df_box['Metric'].unique()
    n_metrics = len(unique_metrics)
    for i in range(n_metrics - 1):
        plt.axvline(x=i + 0.5, color='white', linestyle='--', linewidth=1)

    plt.tight_layout()

    # Save boxplot figure to output_dir_figures as a PNG file
    boxplot_file = os.path.join(output_dir_figures, "boxplot_metrics.png")
    plt.savefig(boxplot_file, dpi=fig_dpi)
    plt.close()

    # -----------------------------------------------------------------------------------
    #
    # Bar Plot
    #
    barplot_data = []
    for phase in ['train', 'val']:
        for metric, stats in aggregated_metrics[phase].items():
            # Skip the "run" key because it's not a metric.
            if metric in ("run", "loss"):
                continue
            metric_label = metric.replace("_", " ").title()
            barplot_data.append({
                'Phase': phase_mapping[phase],
                'Metric': metric_label,
                'Mean': stats['mean'],
                'Std': stats['std']
            })
    df_bar = pd.DataFrame(barplot_data)

    # Pivot the DataFrame so that each row is a metric with separate columns for each phase.
    df_pivot = df_bar.pivot(index='Metric', columns='Phase', values=['Mean', 'Std'])
    mean_df = df_pivot['Mean']
    std_df = df_pivot['Std']

    # 1) Define the new ordering
    ordered_metrics = []
    # First the two accuracy metrics
    for key in ['Per Label Accuracy', 'Per Image Accuracy']:
        if key in mean_df.index:
            ordered_metrics.append(key)
    # Then all other metrics except Loss
    for m in mean_df.index:
        if m not in ordered_metrics and m != 'Loss':
            ordered_metrics.append(m)
    # Finally, append Loss
    if 'Loss' in mean_df.index:
        ordered_metrics.append('Loss')

    # 2) Reindex the DataFrames
    mean_df = mean_df.reindex(ordered_metrics)
    std_df  = std_df.reindex(ordered_metrics)
    metrics = ordered_metrics

    # 3) Extract the values in the new order
    train_means = mean_df['Training'].values
    train_std   = std_df['Training'].values
    val_means   = mean_df['Validation'].values
    val_std     = std_df['Validation'].values

    # 4) Plot
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, train_means, width, yerr=train_std, capsize=5, label='Training')
    ax.bar(x + width / 2, val_means, width, yerr=val_std, capsize=5, label='Validation')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0)
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Mean Value", fontweight="bold")
    ax.set_title(f"{fig_title_prefix} | {model_out_prefix.upper()} Metrics | {n_runs} Runs and {k_folds} Folds")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Save bar plot figure to output_dir_figures as a PNG file
    barplot_file = os.path.join(output_dir_figures, "barplot_metrics.png")
    plt.savefig(barplot_file, dpi=fig_dpi)
    plt.close()

    # -----------------------------------------------------------------------------------
    #
    # Acc vs Loss Plots
    #
    df_train = pd.DataFrame({
        'Train Loss': global_metrics_history['train']['loss'],
        'Per Label Accuracy': global_metrics_history['train']['per_label_accuracy'],
        'Per Image Accuracy': global_metrics_history['train']['per_image_accuracy'],
        'Run': global_metrics_history['train']['run']
    })

    df_val = pd.DataFrame({
        'Validation Loss': global_metrics_history['val']['loss'],
        'Per Label Accuracy': global_metrics_history['val']['per_label_accuracy'],
        'Per Image Accuracy': global_metrics_history['val']['per_image_accuracy'],
        'Run': global_metrics_history['val']['run']
    })

    # Create a figure with 2 rows and 2 columns.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    # -------------------------
    # Top Left: Training - Per Label Accuracy vs Train Loss
    sns.scatterplot(ax=axes[0, 0], data=df_train, x='Train Loss', y='Per Label Accuracy', hue='Run', s=100)
    axes[0, 0].set_title("Training: Per Label Accuracy vs Train Loss")
    axes[0, 0].set_xlabel("Train Loss")
    axes[0, 0].set_ylabel("Per Label Accuracy")

    # -------------------------
    # Top Right: Training - Per Image Accuracy vs Train Loss
    sns.scatterplot(ax=axes[0, 1], data=df_train, x='Train Loss', y='Per Image Accuracy', hue='Run', s=100)
    axes[0, 1].set_title("Training: Per Image Accuracy vs Train Loss")
    axes[0, 1].set_xlabel("Train Loss")
    axes[0, 1].set_ylabel("Per Image Accuracy")

    # -------------------------
    # Bottom Left: Validation - Per Label Accuracy vs Validation Loss
    sns.scatterplot(ax=axes[1, 0], data=df_val, x='Validation Loss', y='Per Label Accuracy', hue='Run', s=100, marker='^')
    axes[1, 0].set_title("Validation: Per Label Accuracy vs Validation Loss")
    axes[1, 0].set_xlabel("Validation Loss")
    axes[1, 0].set_ylabel("Per Label Accuracy")

    # -------------------------
    # Bottom Right: Validation - Per Image Accuracy vs Validation Loss
    sns.scatterplot(ax=axes[1, 1], data=df_val, x='Validation Loss', y='Per Image Accuracy', hue='Run', s=100, marker='^')
    axes[1, 1].set_title("Validation: Per Image Accuracy vs Validation Loss")
    axes[1, 1].set_xlabel("Validation Loss")
    axes[1, 1].set_ylabel("Per Image Accuracy")

    fig.suptitle(
        f"{fig_title_prefix} | {model_out_prefix.upper()} Acc vs Loss "
        f"| {n_runs} Runs and {k_folds} Folds",
        fontsize=13,
        y=0.99
    )

    # Adjust layout and display legend
    plt.tight_layout()

    # Save the figure
    scatterplot_file = os.path.join(output_dir_figures, "scatter_acc_vs_loss.png")
    plt.savefig(scatterplot_file, dpi=fig_dpi)
    plt.close()

    # -----------------------------------------------------------------------------------
    #
    # Learning Curves (Acc vs Loss)
    #
    # Build the DataFrame from 'epoch_metrics_data'
    df_epoch = pd.DataFrame(epoch_metrics_data)

    # Create a figure with three vertically stacked panels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    epochs = sorted(df_epoch['Epoch'].unique())

    colors     = {'Training': 'blue', 'Validation': 'orange'}
    markers    = {'Training': 'o', 'Validation': '^'}
    linestyles = {'Training': '-', 'Validation': '-'}

    # --------------------- Panel 1: Per-Label Accuracy ---------------------
    for (run, fold, phase), group in df_epoch.groupby(['Run', 'Fold', 'Phase']):
        g = group.sort_values('Epoch')
        ax1.plot(
            g['Epoch'], g['per_label_accuracy'],
            color=colors[phase],
            linestyle=linestyles[phase],
            marker=markers[phase],
            alpha=0.6,
            label=f"{phase} (Run {run}, Fold {fold})"
        )

    ax1.set_ylim(0, 100)
    ax1.set_title("Per-Label Accuracy over Epochs")
    ax1.set_ylabel("Per-Label Accuracy (%)")

    # --------------------- Panel 2: Per-Image Accuracy ---------------------
    for (run, fold, phase), group in df_epoch.groupby(['Run', 'Fold', 'Phase']):
        g = group.sort_values('Epoch')
        ax2.plot(
            g['Epoch'], g['per_image_accuracy'],
            color=colors[phase],
            linestyle=linestyles[phase],
            marker=markers[phase],
            alpha=0.6,
            label=f"{phase} (Run {run}, Fold {fold})"
        )

    ax2.set_ylim(0, 100)
    ax2.set_title("Per-Image Accuracy over Epochs")
    ax2.set_ylabel("Per-Image Accuracy (%)")

    # --------------------- Panel 3: Loss ---------------------
    for (run, fold, phase), group in df_epoch.groupby(['Run', 'Fold', 'Phase']):
        g = group.sort_values('Epoch')
        ax3.plot(
            g['Epoch'], g['Loss'],
            color=colors[phase],
            linestyle='--',
            marker=markers[phase],
            alpha=0.6,
            label=f"{phase} (Run {run}, Fold {fold})"
        )

    # Set Loss y‑limit relative to first‑epoch loss
    initial_loss = df_epoch[df_epoch['Epoch'] == 1]['Loss'].min()
    ax3.set_ylim(0, initial_loss * 1.5)
    ax3.set_title("Loss over Epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")

    # Then, after all the plotting but before tight_layout:
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(epochs)

    # Create custom handles for the two phases
    legend_handles = [
        Line2D([0], [0], color=colors['Training'], linewidth=2, label='Training'),
        Line2D([0], [0], color=colors['Validation'], linewidth=2, label='Validation')
    ]

    # Make room at the top and draw the legend centered
    fig.subplots_adjust(top=0.92)
    fig.legend(handles=legend_handles,
               loc='upper center',
               ncol=2,
               frameon=False,
               bbox_to_anchor=(0.915, 1.0))

    fig.suptitle(
        f"{fig_title_prefix} | {model_out_prefix.upper()} Learning Curves "
        f"| {n_runs} Runs and {k_folds} Folds",
        fontsize=14,
        y=0.99
    )

    plt.tight_layout()
    plt.show()

    # Save the combined learning curve figure.
    learning_curve_file = os.path.join(output_dir_figures, "learning_curves.png")
    plt.savefig(learning_curve_file, dpi=fig_dpi)
    plt.close()


    # ----------------------------------------------- End of Main -----------------------------------------------------
    #

    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}] ❯❯❯ Model Training Complete ❮❮❮")
    print(f"[{get_print_time()}]")
    end_time_overall = time.time()
    overall_runtime = end_time_overall - start_time_overall
    print(f"[{get_print_time()}] Overall Runtime {timedelta(seconds=overall_runtime)}")
    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}]")

# ---------------------------------------------------------------------------------------------------------------------
#
if __name__ == '__main__':
    main(sys.argv[1:])
