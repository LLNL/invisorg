
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

from enum import Enum

# ----------------------------------------------------- Enums ---------------------------------------------------------
#
#
class ClassLabel(Enum):
    """
    Enumeration used the dataset and dataloader classes for identifying the target column(s) to assign a sample.
    """
    DISEASE_STATUS = "status"    # Binary Classification.
    DISEASE_TYPE = "type"        # Multi-Class Classification.
    BODY_SITE = "site"           # Multi-Class Classification.
    COUNTRY = "country"          # Multi-Class Classification.
    MULTI_LABEL = 'multi_label'  # Multi-Label Classification with all of the above.

    @property
    def mapping(self) -> tuple[str, str]:
        """
        Returns the (<value>_unique_labels, <value>_enum) tuple.
        """
        base = self.value
        return (f"{base}_unique_labels", f"{base}_enum")


# --------------------------------------------------- Classes ---------------------------------------------------------
#
#
class MicrobiomeMapNormalize(object):
    """Specialized normalization for sparse Microbiome Maps"""
    def __init__(self, threshold=0.01, clip_value=3.0):
        self.threshold = threshold
        self.clip_value = clip_value

    def __call__(self, tensor):
        tensor = tensor.clone()

        # First, log-transform to reduce extreme values, and add small epsilon to avoid log(0)
        tensor = torch.log1p(tensor * 255.0) / 5.5  # log1p(255) ≈ 5.5

        # Sparse normalization
        for c in range(tensor.shape[0]):
            channel = tensor[c]
            non_zero_mask = channel > self.threshold

            if non_zero_mask.any():
                non_zero_pixels = channel[non_zero_mask]
                mean = non_zero_pixels.mean()
                std = non_zero_pixels.std() + 1e-8

                # Normalize and clip
                channel[non_zero_mask] = (non_zero_pixels - mean) / std
                channel[non_zero_mask] = torch.clamp(
                    channel[non_zero_mask],
                    -self.clip_value,
                    self.clip_value
                )

        return tensor


# ---------------------------------------------------------------------------------------------------------------------
#
#
class PublicMicrobiomeDataset(Dataset):
    """
    Dataset class for the public data.
    """

    def load_csv(self, path):
        """
        Loads a CSV file from the file system. The CSV file will have the header:
            sample_name,dataset,status,type,site,country

        :param path: The path to the CSV file.
        :return: A Pandas DataFrame with the contents of the CSV file.
        """
        df_abundance_train = pd.read_csv(path, sep=',', header=0, index_col=False)

        return df_abundance_train


    def __init__(self, csv_file, root_dir, target_factor, class_support:int = 20, should_augment:bool = False,
                 height_width:int = 256, transform=None, disease_only:bool = False, use_hfe: bool = False,
                 hfe_features: str = None):
        """
        Initialize the dataset for image-based microbiome classification tasks.

        Parameters
        ----------
        csv_file        : str Path to the CSV file containing sample metadata and target labels.
        root_dir        : str Directory where the corresponding images are stored.
        target_factor   : str Which label to train against (e.g. "disease_status", "disease_type", "body_site",
                            "country").
        class_support   : int optional. Minimum number of samples required per class; classes with fewer
                        samples will be filtered out (default=20).
        should_augment  : bool, optional. If True, apply data augmentation to training images (default=False).
        height_width    : int, optional Size (in pixels) to which each image will be resized or center-cropped
                            (height_width x height_width) (default=256).
        transform       : callable, optional. A torchvision (or custom) transform to apply to each image after
                            resizing/cropping (default=None).
        disease_only    : bool, optional. If True, only use samples with disease_status='diseased' when
                            target_factor is DISEASE_TYPE or MULTI_LABEL (default=False).
        use_hfe         : bool, optional. If True, load and use HFE (High-level Feature Embeddings) features
                            for each sample (default=False).
        hfe_features    : str, optional. Path to the NPZ file containing HFE learned embeddings. Required when
                            use_hfe=True. The NPZ file must contain 'sample_ids', 'embeddings', and 'labels'
                            arrays as created by np.savez_compressed() (default=None).
        """

        super().__init__()

        self.csv_file          = csv_file
        self.root_dir          = root_dir
        self.transform         = transform
        self.should_augment    = should_augment
        self.height_width      = height_width
        self.target_factor     = target_factor
        self.use_hfe           = use_hfe
        # Normalize “None” to False so that only `True` trips the HFE loader.
        self.use_hfe = bool(self.use_hfe)

        # Determine which target factor to use
        self.dims = None
        if target_factor == ClassLabel.DISEASE_STATUS:
            self.dims = [ClassLabel.DISEASE_STATUS.value]
        elif target_factor == ClassLabel.DISEASE_TYPE:
            self.dims = [ClassLabel.DISEASE_TYPE.value]
        elif target_factor == ClassLabel.COUNTRY:
            self.dims = [ClassLabel.COUNTRY.value]
        elif target_factor == ClassLabel.BODY_SITE:
            self.dims = [ClassLabel.BODY_SITE.value]
        elif target_factor == ClassLabel.MULTI_LABEL:
            self.dims = [
                ClassLabel.DISEASE_STATUS.value,
                ClassLabel.DISEASE_TYPE.value,
                ClassLabel.BODY_SITE.value,
                ClassLabel.COUNTRY.value,
            ]
        assert self.dims is not None, '[ERROR] self.dims is NONE.'

        # 1) Load the full table of data.
        self.df_abundances = self.load_csv(path=self.csv_file)

        # Filter for diseased samples, but only if requested
        if disease_only and target_factor in (ClassLabel.DISEASE_TYPE, ClassLabel.MULTI_LABEL):
            diseased_mask = self.df_abundances[ClassLabel.DISEASE_STATUS.value] == 'diseased'
            self.df_abundances = self.df_abundances[diseased_mask].reset_index(drop=True)

        # 2) Drop low-support classes based on selected dims. We look for the header in the CSV.
        #    These should match: "sample_name,dataset,status,type,site,country"
        for col in self.dims:
            counts = self.df_abundances[col].value_counts()
            low_support = counts[counts <= class_support].index
            if len(low_support):
                self.df_abundances = (
                    self.df_abundances
                        .loc[~self.df_abundances[col].isin(low_support)]
                        .reset_index(drop=True)
                )

        # 3) Balanced augmentation across selected factors (dims).
        self.df_abundances['_augment'] = False
        if self.should_augment:
            aug_rows = []
            for col in self.dims:
                counts = self.df_abundances[col].value_counts()
                max_count = counts.max()
                for label, cnt in counts.items():
                    needed = max_count - cnt
                    if needed <= 0:
                        continue
                    candidates = self.df_abundances[self.df_abundances[col] == label]
                    for _ in range(needed):
                        row = candidates.sample(n=1, replace=True).iloc[0].to_dict()
                        row['_augment'] = True
                        aug_rows.append(row)
            if aug_rows:
                self.df_abundances = pd.concat(
                    [self.df_abundances, pd.DataFrame(aug_rows)], ignore_index=True
                )

        # 4) Factorize label columns to enumerations (selected dims & fit binarizers)
        #    We encode the columns as enumerated types using pd.factorize() which returns:
        #       1. codes: an integer ndarray that’s an indexer into 'uniques'.
        #       2. uniques: The unique valid values (1-D ndarray).
        self.col_to_mlb = {}

        for col in self.dims:
            uniques_name, enum_name = ClassLabel(col).mapping
            codes, uniques = pd.factorize(self.df_abundances[col])

            setattr(self, uniques_name, list(uniques))
            self.df_abundances[enum_name] = codes

            # fit MultiLabelBinarizer
            mlb = MultiLabelBinarizer(classes=getattr(self, uniques_name))
            mlb.fit([[u] for u in getattr(self, uniques_name)])
            self.col_to_mlb[col] = mlb

        # 5) Compute total multi-label dimensions efficiently
        self.mln_classes = sum(len(mlb.classes_) for mlb in self.col_to_mlb.values())
        self.mln_class_names = [c for mlb in self.col_to_mlb.values() for c in mlb.classes_]

        # 7) Default transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            MicrobiomeMapNormalize()
        ])
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            MicrobiomeMapNormalize()
        ])

        # 8) Load HFE features if requested
        if self.use_hfe:
            if hfe_features is None:
                raise ValueError("hfe_features must be provided when use_hfe is True")
            # Load NPZ file with HFE features
            npz_data = np.load(hfe_features, allow_pickle=True)
            sample_ids = npz_data['sample_ids']
            embeddings = npz_data['embeddings']
            # Create a mapping from sample_id to embedding index for fast lookup
            self.hfe_id_to_idx = {str(sid): idx for idx, sid in enumerate(sample_ids)}
            self.hfe_embeddings = embeddings


    def __len__(self):
        """
        Return the number of samples in the dataset.

        This method calculates the total number of samples by returning the length of the index
        of the 'df_abundances' DataFrame, which represents the dataset's entries.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.df_abundances.index)


    def add_gaussian_noise(self, img_tensor, mean=0.0, std=0.1):
        """
        Adds Gaussian noise to an image tensor. This method generates random Gaussian noise with a
        specified mean and standard deviation and adds it to the input image tensor. The resulting
        image tensor is then clipped to ensure pixel values remain within the valid range of [0, 255].

        Args:
            img_tensor (torch.Tensor): The input image tensor to which noise will be added.
                                    Expected shape is [C, H, W], where C is the number of channels,
                                    and H and W are the height and width of the image.
            mean (float, optional): The mean of the Gaussian noise. Default is 0.0.
            std (float, optional): The standard deviation of the Gaussian noise. Default is 0.1.

        Returns:
            torch.Tensor: The noisy image tensor with the same shape as the input tensor, where noise
                        has been added and values are clipped to the range [0, 255].
        """

        noise = torch.randn_like(img_tensor) * std + mean

        return torch.clamp(img_tensor + noise, 0.0, 1.0)


    def __getitem__(self, idx):
        """
        Retrieve and process a dataset sample at the given index. This method extracts various label and class
        information from the 'df_abundances' DataFrame based on the provided index, including:
        - A human-readable "label" and its enumerated "class".
        - Disease status, category, body site, and country labels along with their corresponding
            enumerated classes.
        - Multi-label targets generated using multi-label binarizers (mlb_status, mlb_category,
            mlb_body_site, and mlb_country).

        The method then constructs the image path using the 'Sample Name' from the DataFrame and attempts to
        load the corresponding '.png' image file using torchvision. If the image file is found, it is read,
        converted to a float tensor, and incorporated into a dictionary along with the label and class information.

        This method performs the following steps:
            1. Extracts the sample's label and class information from the underlying DataFrame.
            2. Retrieves categorical information for multiple label dimensions (e.g., Disease Status, Disease Category,
                Body Site, Country) along with their corresponding enumerated classes.
            3. Uses pre-fitted MultiLabelBinarizers to transform each categorical label into a binary vector
                without re-fitting.
            4. Concatenates the individual binary vectors into a single multi-label target tensor.
            5. Constructs the image file path based on the sample's name, loads the image as an RGB tensor using
                torchvision and converts it to float.
            6. Optionally applies a provided transformation to the sample.
            7. Returns a dictionary containing both the image data and the associated label and class metadata.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - 'id': The sample identifier (e.g., file name).
                - 'image': A float tensor representing the RGB image.
                - 'image_path': The full path to the image file.
                - 'label': The human-readable label from the target column.
                - 'class': The enumerated class corresponding to the label.
                - 'label_status': The disease status label.
                - 'class_status': The enumerated disease status.
                - 'label_category': The disease category label.
                - 'class_category': The enumerated disease category.
                - 'label_body_site': The body site label.
                - 'class_body_site': The enumerated body site.
                - 'label_country': The country label.
                - 'class_country': The enumerated country.
                - 'multi_label_target': A concatenated float tensor representing the multi-label binary target.

        """
        row = self.df_abundances.iloc[idx]

        if self.target_factor is ClassLabel.MULTI_LABEL:
            # no single “label” or “class” column for multi-label tasks
            label     = ""
            img_class = -1
        else:
            label     = row[self.target_factor.value]
            img_class = int(row[self.target_factor.mapping[1]])

        # Build target based on selected factors
        # We use transform() instead of fit_transform() to avoid re-fitting the binarizer
        targets = []
        for col in self.dims:
            mlb = self.col_to_mlb[col]
            vec = mlb.transform([[row[col]]])[0]
            targets.append(torch.tensor(vec, dtype=torch.float))
        target = torch.cat(targets)

        # Load image
        img_name = row['sample_name']
        img_path = f"{self.root_dir}/{img_name}.png"
        try:
            img_tensor = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        except (FileNotFoundError, RuntimeError):
            print(f"Image file not found, skipping: {img_path}")
            return self.__getitem__(idx + 1)

        pil_img = transforms.ToPILImage()(img_tensor)
        if self.should_augment and row['_augment']:
            img_tensor = self.aug_transform(pil_img)
        else:
            img_tensor = self.base_transform(pil_img)

        sample = {
            'id': img_name,
            'image': img_tensor,
            'image_path': img_path,
            'label': label,
            'class': img_class,
            # include all enums for metadata
            'label_status': row.get(ClassLabel.DISEASE_STATUS.value),
            'class_status': int(row.get(ClassLabel.DISEASE_STATUS.mapping[1], -1)),
            'label_category': row.get(ClassLabel.DISEASE_TYPE.value),
            'class_category': int(row.get(ClassLabel.DISEASE_TYPE.mapping[1], -1)),
            'label_body_site': row.get(ClassLabel.BODY_SITE.value),
            'class_body_site': int(row.get(ClassLabel.BODY_SITE.mapping[1], -1)),
            'label_country': row.get(ClassLabel.COUNTRY.value),
            'class_country': int(row.get(ClassLabel.COUNTRY.mapping[1], -1)),
            'target': target,
        }

        if self.use_hfe:
            try:
                # Use index lookup instead of pandas loc
                hfe_idx = self.hfe_id_to_idx[img_name]
                features = self.hfe_embeddings[hfe_idx]
            except KeyError:
                raise KeyError(f"HFE features not found for sample_id {img_name}")
                # features = np.random.rand(512)

            sample["hfe_features"] = torch.tensor(features, dtype=torch.float)

        return sample
