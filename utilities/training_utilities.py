"""
Miscellaneous classes and functions used in the training script.
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
from utilities.basic_utilities import get_print_time
from datasets.dataset import ClassLabel
from utilities.model_classes import ModelTypeEnum
from models.cnn import MmSparseAwareCNN
from models.vit import MmViT

np.random.seed(0)
torch.manual_seed(0)

sns.set_theme(style="darkgrid")
sns.set_context("paper")


# ---------------------------------------------- Functions & Methods --------------------------------------------------
#
# region Functions & Methods

def print_one_fold_message():
    """
    Print a warning message to STDOUT with the word 'WARNING' in red.
    The message states that we are running the full program, but only using one fold of data.
    This is for testing purposes and slightly different than develop mode.

    Note: \033[91m starts red text; \033[0m resets the color.
    """
    print(f"[{get_print_time()}] ⚠️  \033[91mWARNING: One Fold!\033[0m ⚠️")


def to_categorical(y, num_classes):
    """
    1-hot encodes a tensor. Based on Keras 'keras.utils.to_categorical' which converts a class vector (integers)
    to binary class matrix.
    Uses numpy's eye() function which returns a 2-D array with ones on the diagonal and zeros elsewhere.
    """
    return np.eye(num_classes, dtype='uint8')[y]


def flatten(nested_list):
    """
    Flattens a nested list of lists.

    :param nested_list: A python nested List of lists.
    :return: A flattened list (no nested lists).
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def unwrap_subset(ds):
    """
    Recursively unwraps a (possibly nested) Subset object to retrieve the base dataset and the combined indices.

    If the input dataset `ds` is an instance of Subset (or nested Subset), this function traverses through
    each layer to access the original dataset (which is expected to have attributes like `df_abundances`)
    and combines the indices from each Subset layer so that they correctly index into the base dataset.
    If `ds` is not a Subset, it returns the dataset itself with indices set to None.

    Args:
        ds: A dataset or Subset object which may wrap other Subset objects.

    Returns:
        tuple: A tuple (base_dataset, combined_indices) where:
            - base_dataset: The original dataset object (with attributes like `df_abundances`).
            - combined_indices: A list of indices mapping to rows in the base_dataset corresponding to the
              nested Subset selection. If `ds` is not a Subset, returns None for combined_indices.
    """
    indices = None
    while hasattr(ds, 'dataset'):
        if indices is None:
            indices = ds.indices
        else:
            # Map the current indices through the new subset's indices
            indices = [ds.indices[i] for i in indices]
        ds = ds.dataset
    return ds, indices


def get_summary_df(ds):
    """
    Retrieves a summary DataFrame with label and class information from a dataset or Subset.

    This function unwraps nested Subset objects to access the underlying dataset and its DataFrame
    (expected to be stored in the `df_abundances` attribute). It then filters the DataFrame to include
    only the rows corresponding to the indices present in the subset. The function extracts the
    'Sample Name', the target column (which contains the human-readable label), and 'Enumerated' (which
    corresponds to the class ID). Finally, it renames the target column to 'label' and 'Enumerated' to 'class'
    for standardization.

    Args:
        ds: A dataset or Subset object where the underlying dataset contains a DataFrame in `df_abundances`
            and attributes such as `target_col`.

    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - 'Sample Name': The sample identifier.
            - 'label': The human-readable label extracted from the target column.
            - 'class': The enumerated class corresponding to the label.
    """
    base_ds, indices = unwrap_subset(ds)
    df = base_ds.df_abundances
    if indices is not None:
        df = df.iloc[indices]

    # Multi-label: one row per (sample, dimension)
    if base_ds.target_factor is ClassLabel.MULTI_LABEL:
        summary_parts = []
        for col in base_ds.dims:
            # mapping gives (<value>_unique_labels, <value>_enum)
            _, enum_name = ClassLabel(col).mapping
            part = df[['sample_name', col, enum_name]].rename(
                columns={col: 'label', enum_name: 'class'}
            )
            summary_parts.append(part)
        return pd.concat(summary_parts, ignore_index=True)

    # Single-label (binary or multi-class)
    enumeration_name = base_ds.target_factor.mapping[1]
    return df[['sample_name', base_ds.target_factor.value, enumeration_name]].rename(
        columns={base_ds.target_factor.value: 'label', enumeration_name: 'class'}
    )


def print_label_summary(ds_train, ds_val, tablefmt="github"):
    """
    Prints a consolidated table summarizing label counts and corresponding class IDs for training
    and validation data sets. This function uses 'get_summary_df()' to retrieve summary "DataFrames"
    for both the training and validation splits. It then groups each DataFrame by the label to compute:
      - The count of samples per label.
      - The first encountered class ID for each label.

    The grouped summaries from both splits are merged into a single table and printed using the specified
    table format.

    Args:
        ds_train (Dataset or Subset): The training dataset (or Subset) containing samples with label and class information.
        ds_val (Dataset or Subset): The validation dataset (or Subset) containing samples with label and class information.
        tablefmt (str, optional): The format for printing the table (e.g., "github", "plain", "fancy_grid").
        Defaults to "github".

    Returns:
        None: The function prints the summary table to stdout.
    """

    # Create summary DataFrames for train and validation splits
    df_train = get_summary_df(ds_train)
    df_val   = get_summary_df(ds_val)

    # Compute counts and class mapping per label
    train_counts = df_train.groupby('label').agg({'sample_name': 'count', 'class': 'first'}).rename(
        columns={'sample_name': 'Train Count', 'class': 'Train Class'})

    val_counts   = df_val.groupby('label').agg({'sample_name': 'count', 'class': 'first'}).rename(
        columns={'sample_name': 'Val Count', 'class': 'Val Class'})

    # Merge the two summaries
    summary_msg = pd.concat([train_counts, val_counts], axis=1).fillna('')
    summary_msg = summary_msg.reset_index().rename(columns={'index': 'Label'})

    print()
    print(tabulate(summary_msg, headers="keys", tablefmt=tablefmt))
    print()


def compute_metrics(y_true, y_pred, is_multi_label=True, threshold=0.5):
    """
    Compute classification metrics including accuracy, precision, recall, and F1 score.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_labels)
        Ground truth binary labels.
    y_pred : array-like, shape (n_samples, n_labels) or probabilities
        Predicted labels or probabilities.
    is_multi_label : bool, default True
        Flag indicating that the task is multi-label classification.
    threshold : float, default 0.5
        Threshold used to convert predicted probabilities to binary labels.

    Returns
    -------
    A dictionary containing:
        'per_label_accuracy' : float
            Fraction of individual label predictions that are correct.
        'per_image_accuracy' : float
            Fraction of samples where every label is predicted correctly.
        'precision_macro'    : float
            Macro-averaged precision.
        'recall_macro'       : float
            Macro-averaged recall.
        'f1_macro'           : float
            Macro-averaged F1 score.
        'precision_samples'  : float
            Sample-averaged precision.
        'recall_samples'     : float
            Sample-averaged recall.
        'f1_samples'         : float
            Sample-averaged F1 score.
    """

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_flat = None
    y_pred_flat = None

    if is_multi_label:
        # Binarize predictions if they are not already binary.
        y_pred_bin = np.where(y_pred >= threshold, 1, 0)

        # Per-label accuracy: fraction of individual labels correctly predicted.
        per_label_accuracy = np.mean(y_true == y_pred_bin)

        # Per-image accuracy: fraction of samples for which every label is correct.
        per_image_accuracy = np.mean(np.all(y_true == y_pred_bin, axis=1))

        # Re-assign so that we can have one path below, and flatten for per‑sample (samples‑average) metrics
        y_true_flat = y_true
        y_pred_flat = y_pred_bin

    else:
        # Single-label: if y_pred has >1 column, assume class probabilities/logits
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Convert to predicted class indices
            y_pred_bin = y_pred.argmax(axis=1)
        else:
            # Binary or already integer labels
            y_pred_bin = y_pred.ravel()

        # True labels should be integer class indices
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred_bin.ravel()

        # Per-label and per-image accuracy collapse to the same thing
        per_label_accuracy = np.mean(y_true_flat == y_pred_flat)
        per_image_accuracy = per_label_accuracy

    assert y_true_flat is not None, '[ERROR] compute_metrics function: y_true_flat is NONE.'
    assert y_pred_flat is not None, '[ERROR] compute_metrics function: y_pred_flat is NONE.'

    # Compute metrics using scikit-learn's functions.
    precision_macro = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall_macro    = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    f1_macro        = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)

    if is_multi_label:
        precision_samples = precision_score(y_true_flat, y_pred_flat, average='samples', zero_division=0)
        recall_samples    = recall_score(y_true_flat, y_pred_flat, average='samples', zero_division=0)
        f1_samples        = f1_score(y_true_flat, y_pred_flat, average='samples', zero_division=0)
    else:
        # for single-label, sample-wise metrics collapse to overall accuracy
        precision_samples = per_image_accuracy
        recall_samples    = per_image_accuracy
        f1_samples        = per_image_accuracy

    return {
        'per_label_accuracy': per_label_accuracy,
        'per_image_accuracy': per_image_accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_samples': precision_samples,
        'recall_samples': recall_samples,
        'f1_samples': f1_samples
    }


def get_true_labels(dataloader, device, is_multi_label):
    """
    Extract ground truth labels from a dataloader.

    Args:
        dataloader: The DataLoader for the test set.
        device: The torch device.
        is_multi_label (bool): If True, use 'multi_label_target', else 'class'.

    Returns:
        A torch.Tensor of concatenated labels.
    """
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            if is_multi_label:
                labels = batch['target']
            else:
                labels = batch['class']
            all_labels.append(labels.cpu())

    return torch.cat(all_labels, dim=0)


def predict_with_model(model, dataloader, device):
    """
    Generate predictions for all batches in a DataLoader using the given model.

    This function iterates over the provided dataloader, moves the input images to the
    specified device, and passes them through the model to obtain predictions. The outputs
    are moved to the CPU and concatenated into a single tensor.

    Args:
        model (torch.nn.Module): The PyTorch model used for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader that yields batches of data.
            Each batch should be a dictionary with an 'image' key containing the input images.
        device (torch.device): The device (CPU or GPU) on which to perform inference.

    Returns:
        torch.Tensor: A tensor containing the concatenated predictions from all batches.
                      The shape will depend on the model's output format.
    """
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            hfe = batch.get('hfe_features')
            if hfe is not None:
                hfe = hfe.to(device)

            model_to_check = model.module if isinstance(model, nn.DataParallel) else model
            if getattr(model_to_check, "use_hfe", False):
                preds = model(images, hfe_features=hfe)
            else:
                preds = model(images)

            predictions.append(preds.cpu())

    return torch.cat(predictions, dim=0)


def model_type_enum(s):
    """
    Convert a string to a ModelTypeEnum member.

    This function takes an input string representing the model type, normalizes it to lowercase,
    and returns the corresponding ModelTypeEnum member. If the input does not match any of the
    valid options (i.e., 'vit', 'cnn', or 'fusion'), an argparse.ArgumentTypeError is raised.

    Args:
        s (str): The input model type as a string.

    Returns:
        ModelTypeEnum: The corresponding enum member (e.g., ModelTypeEnum.VIT, ModelTypeEnum.CNN, ModelTypeEnum.FUSION).

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid model type.
    """
    try:
        return ModelTypeEnum(s.lower())
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid model type. Choose from: {}.".format(
            ", ".join([e.value for e in ModelTypeEnum])
        ))


def get_progressive_color_transform(current_epoch, total_epochs):
    max_brightness = 0.5
    max_contrast = 0.5
    max_saturation = 0.5
    max_hue = 0.1

    # Progress factor (0.0 to 1.0)
    factor = current_epoch / total_epochs

    return transforms.Compose([
        transforms.ColorJitter(
            brightness=factor * max_brightness,
            contrast=factor * max_contrast,
            saturation=factor * max_saturation,
            hue=factor * max_hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_model(model_class, model_path, device, withDataParallel=True, freeze=False, **model_kwargs):
    """
    Load a model from a saved state_dict, optionally handling DataParallel prefixes and freezing parameters.

    Args:
        model_class: The class of the model to instantiate.
        model_path: Path to the saved model (.pth file).
        device: Torch device to load the model to.
        withDataParallel (bool): If True, remove the "module." prefix from state_dict keys.
        freeze (bool): If True, freeze model parameters (disable gradient updates).
        **model_kwargs: Additional keyword arguments required to instantiate the model.

    Returns:
        The loaded model, set to evaluation mode.
    """
    model = model_class(**model_kwargs)
    state_dict = torch.load(model_path, map_location=device)

    # Check if "module." prefix exists in keys
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if withDataParallel and has_module_prefix:
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.to(device)
    model.eval()
    return model


def load_backbones_for_fusion(vit_models_dir, cnn_models_dir, device, is_multi_label, n_channels, height_width,
                              vit_number_patches, vit_number_blocks, vit_hidden_d, vit_number_heads,
                              cnn_hidden_d, num_classes, is_binary_class, freeze_backbones=False):
    """
    Recursively loads pretrained ViT and CNN backbones from specified directory trees, strips their classification heads,
    and returns lists of ViT and CNN models ready for late fusion.

    Args:
        vit_models_dir (str): Base directory containing ViT model subdirectories (e.g., run_1/, run_2/).
        cnn_models_dir (str): Base directory containing CNN model subdirectories.
        device (torch.device): Device to load models onto ("cpu", "cuda", or "mps").
        is_multi_label (bool): Whether the task is multi-label classification.
        n_channels (int): Number of input channels.
        height_width (int): Height/Width of the input images (assumed square).
        vit_number_patches (int): Number of patches to split images into (ViT).
        vit_number_blocks (int): Number of transformer encoder blocks (ViT).
        vit_hidden_d (int): Hidden dimension size for ViT.
        vit_number_heads (int): Number of attention heads for ViT.
        cnn_hidden_d (int): Hidden dimension size for CNN.
        num_classes (int): Number of output classes.
        is_binary_class (bool): Whether the task is binary classification.
        freeze_backbones (bool): If True, backbone models are frozen (no training updates).

    Returns:
        tuple:
            - vit_models (list): List of ViT models (backbone only).
            - cnn_models (list): List of CNN models (backbone only).
    """

    # Recursively find all .pth files
    vit_model_paths = sorted(glob.glob(os.path.join(vit_models_dir, "**", "*.pth"), recursive=True))
    cnn_model_paths = sorted(glob.glob(os.path.join(cnn_models_dir, "**", "*.pth"), recursive=True))

    vit_models = []
    cnn_models = []

    for vit_path in vit_model_paths:
        vit = load_model(MmViT, vit_path, device, withDataParallel=True, freeze=freeze_backbones,
                         chw=(n_channels, height_width, height_width),
                         n_patches=vit_number_patches,
                         n_blocks=vit_number_blocks,
                         hidden_d=vit_hidden_d,
                         n_heads=vit_number_heads,
                         num_classes=num_classes,
                         is_binary_class=is_binary_class,
                         is_multi_label=is_multi_label)
        vit.mlp = nn.Identity()
        vit_models.append(vit)

    for cnn_path in cnn_model_paths:
        cnn = load_model(MmSparseAwareCNN, cnn_path, device, withDataParallel=True, freeze=freeze_backbones,
                         num_classes=num_classes,
                         input_size=(n_channels, height_width, height_width),
                         hidden_d=cnn_hidden_d,
                         is_multi_label=is_multi_label)
        cnn.classifier = nn.Identity()
        cnn_models.append(cnn)

    return vit_models, cnn_models


def is_pretrained_model(model_type):
    """
    Check if the model type is a pretrained model.
    """
    pretrained_types = {
        ModelTypeEnum.RESNET50, ModelTypeEnum.INCEPTION_V3, ModelTypeEnum.ALEXNET,
        ModelTypeEnum.VIT_B_16, ModelTypeEnum.VIT_B_32, ModelTypeEnum.VIT_L_16
    }
    return model_type in pretrained_types


def is_custom_model(model_type):
    """
    Check if the model type is a custom model.
    """
    custom_types = {ModelTypeEnum.VIT, ModelTypeEnum.CNN, ModelTypeEnum.FUSION}
    return model_type in custom_types


def print_distribution_summary(dataset_name, dataset):
    """
    Print a distribution summary table for disease type vs disease status.

    Args:
        dataset_name (str): Name to display in the header (e.g., "Training", "Test")
        dataset: Dataset or Subset object containing the data
    """
    # Handle both master dataset and Subset objects
    if hasattr(dataset, 'dataset'):
        # It's a Subset, unwrap it
        base_dataset, indices = unwrap_subset(dataset)
    else:
        # It's already the master dataset
        base_dataset = dataset
        indices = None

    # Get the dataframe, filtered to the indices if it's a subset
    df = base_dataset.df_abundances
    if indices is not None:
        df = df.iloc[indices]

    # Check if both dimensions exist
    if (ClassLabel.DISEASE_TYPE.value in base_dataset.dims and
            ClassLabel.DISEASE_STATUS.value in base_dataset.dims):

        print(f"[{get_print_time()}]")
        print(f"[{get_print_time()}] ▸ {dataset_name} Class Distribution Summary:")
        print(f"[{get_print_time()}]")

        # Create crosstab of disease type vs disease status
        df_crosstab = pd.crosstab(
            df[ClassLabel.DISEASE_TYPE.value],
            df[ClassLabel.DISEASE_STATUS.value], margins=False)

        # Calculate total samples per disease type
        df_crosstab['N'] = df_crosstab.sum(axis=1)

        # Convert to DataFrame with proper formatting
        df_display = df_crosstab.reset_index()

        # Title case the disease type column and rename columns properly
        df_display[ClassLabel.DISEASE_TYPE.value] = df_display[ClassLabel.DISEASE_TYPE.value].str.title()
        df_display = df_display.rename(columns={
            ClassLabel.DISEASE_TYPE.value: 'Disease Type',
            **{col: col.title() for col in df_crosstab.columns if col != 'N'}
        })

        # Reorder columns: Disease Type, N, then status columns in alphabetical order
        status_columns = sorted([col for col in df_display.columns if col not in ['Disease Type', 'N']])
        column_order = ['Disease Type', 'N'] + status_columns
        df_display = df_display[column_order]

        # Set 1-based index before adding Total row
        df_display.index = range(1, len(df_display) + 1)

        # Add Total row
        total_row = df_display.select_dtypes(include=[np.number]).sum()
        total_row['Disease Type'] = 'Total'
        df_display = pd.concat([df_display, total_row.to_frame().T], ignore_index=True)
        # Set custom index - 1-based for data, empty for Total
        df_display.index = list(range(1, len(df_display))) + ['']

        print(tabulate(df_display, headers="keys", tablefmt="simple", showindex=True))
        print()


def compute_roc_auc_safe(y_true, y_score, is_multi_label, target_factor):
    """
    Compute ROC-AUC robustly across binary, multi-class, and multi-label tasks.
    - y_true: np.array
        * binary/multi-class: shape (N,) with integer class ids
        * multi-label:        shape (N, C) with 0/1
    - y_score: np.array
        * binary:      shape (N,)           (positive-class probability)
        * multi-class: shape (N, C)         (per-class probabilities)
        * multi-label: shape (N, C)         (per-label probabilities)
    Returns:
        dict with keys:
          'roc_auc_macro'  (float or np.nan)
          'pr_auc_macro'   (float or np.nan)  # average precision (macro)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    def _nanmean_or_nan(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if len(vals) else float('nan')

    # ---------- Binary
    if target_factor is ClassLabel.DISEASE_STATUS and not is_multi_label:
        # y_true: (N,), y_score: (N,)
        try:
            roc_auc = roc_auc_score(y_true.ravel(), y_score.ravel())
        except ValueError:
            roc_auc = float('nan')
        try:
            pr_auc = average_precision_score(y_true.ravel(), y_score.ravel())
        except ValueError:
            pr_auc = float('nan')
        return {'roc_auc_macro': roc_auc, 'pr_auc_macro': pr_auc}

    # ---------- Multi-class
    if not is_multi_label and target_factor in (ClassLabel.DISEASE_TYPE, ClassLabel.BODY_SITE, ClassLabel.COUNTRY):
        # Prefer sklearn's native multi_class handler; if it fails (degenerate class), fallback to per-class OVR.
        try:
            roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        except ValueError:
            # fallback: manual OVR
            C = y_score.shape[1]
            y_onehot = label_binarize(y_true, classes=np.arange(C))
            per_class_auc = []
            per_class_ap  = []
            for c in range(C):
                yt = y_onehot[:, c]
                ys = y_score[:, c]
                # Skip classes with only one label present
                if yt.min() == yt.max():
                    continue
                try:
                    per_class_auc.append(roc_auc_score(yt, ys))
                except ValueError:
                    pass
                try:
                    per_class_ap.append(average_precision_score(yt, ys))
                except ValueError:
                    pass
            roc_auc = _nanmean_or_nan(per_class_auc)
            pr_auc  = _nanmean_or_nan(per_class_ap)
            return {'roc_auc_macro': roc_auc, 'pr_auc_macro': pr_auc}

        # PR-AUC (OVR macro)
        try:
            C = y_score.shape[1]
            y_onehot = label_binarize(y_true, classes=np.arange(C))
            aps = []
            for c in range(C):
                yt = y_onehot[:, c]
                ys = y_score[:, c]
                if yt.min() == yt.max():
                    continue
                aps.append(average_precision_score(yt, ys))
            pr_auc = _nanmean_or_nan(aps)
        except Exception:
            pr_auc = float('nan')
        return {'roc_auc_macro': float(roc_auc), 'pr_auc_macro': pr_auc}

    # ---------- Multi-label
    # y_true: (N, C), y_score: (N, C)
    if is_multi_label:
        C = y_true.shape[1]
        per_label_auc = []
        per_label_ap  = []
        for c in range(C):
            yt = y_true[:, c]
            ys = y_score[:, c]
            # Skip labels with only one class present
            if yt.min() == yt.max():
                continue
            try:
                per_label_auc.append(roc_auc_score(yt, ys))
            except ValueError:
                pass
            try:
                per_label_ap.append(average_precision_score(yt, ys))
            except ValueError:
                pass
        return {
            'roc_auc_macro': _nanmean_or_nan(per_label_auc),
            'pr_auc_macro': _nanmean_or_nan(per_label_ap)
        }

    # Should not reach here
    return {'roc_auc_macro': float('nan'), 'pr_auc_macro': float('nan')}
