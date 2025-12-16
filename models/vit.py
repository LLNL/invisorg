"""
Model class for the microbiome maps Vision Transformer.
    Code is largely based on the "vit_b_16" architecture:
        * https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html
        * https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

"""

import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

# ----------------------------------------------- Utility Functions ---------------------------------------------------
#
# region Patchify
def patchify_simple(images, n_patches):
    """
    Converts a square image into a vector of patches.

    :param images: Image collection.
    :param n_patches: Number of patches that we will find both in width and height.
    """
    batch, channels, height, width = images.shape
    assert height == width, "Patchify only supported for square images (Height == Width)"

    patches = torch.zeros(batch, n_patches ** 2, height * width * channels // n_patches ** 2)
    patch_size = height // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size: (i + 1) * patch_size,
                    j * patch_size: (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()

    return patches


def patchify(images, n_patches):
    """
    Fast batched patchify using unfold.
    """
    batch, channels, height, width = images.shape
    patch_size = height // n_patches
    unfold = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    patches = unfold.transpose(1, 2)
    return patches


# region Positional Embeddings
def get_positional_embeddings(sequence_length, d):
    """
    Transformers, due to their lack of recurrence or convolution, are not capable of learning information about
    the order of a set of tokens. A positional embedding allows the model to understand where each patch would be
    placed in the original image. The positional embedding adds high-frequency values to the first dimensions
    and low-frequency values to the latter dimensions.

    These embeddings converge into vector spaces where they show high similarity to their neighboring
    position embeddings — particularly those sharing the same column and row.

    In each sequence, for token i we add to its j-th coordinate using Sine and Cosine waves (Vaswani et. al).
    See "Attention is All You Need". https://arxiv.org/abs/1706.03762.

    :param sequence_length: _description_
    :param d: _description_
    :return: _description_
    """

    # Positional embeddings are learned vectors with the same dimensionality as our patch embeddings.
    result = torch.ones(sequence_length, d)

    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


# ---------------------------------------------------- Classes --------------------------------------------------------

# ------------------------------------------- Multi-Head Self Attention -----------------------------------------------
#
# region Attention
class MHSA(nn.Module):
    """
    The intuitive idea behind attention is that it allows modeling the relationship between the inputs.
    What makes a '0' a zero are not the individual pixel values, but how they relate to each other.

    Attention is computed from three matrices — Queries, Keys, and Values — each generated from passing the tokens
    through a linear layer.

    We want for a single image, each patch to get updated based on some similarity measure with the other patches.
    We do so by linearly mapping each patch to 3 distinct vectors: q, k, and v (query, key, value).
    For a single patch, we are going to compute the dot product between its q vector with all of the k vectors,
    divide by the square root of the dimensionality of these vectors, softmax these so-called attention cues, and
    finally multiply each attention cue with the v vectors associated with the different k vectors and sum all up.

    Each patch assumes a new value that is based on its similarity (after the linear mapping to q, k, and v) with
    other patches. This whole procedure, however, is carried out H times on H sub-vectors patches, where H is the
    number of Heads.
    """

    def __init__(self, d, n_heads=2, debug_mode: bool = False):
        """
        _summary_

        :param d: _description_
        :param n_heads: _description_, defaults to 2.
        """
        super(MHSA, self).__init__()

        self.d          = d
        self.n_heads    = n_heads
        self.debug_mode = debug_mode

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

        self.attn_temperature = nn.Parameter(torch.ones(self.n_heads))  # per-head scale
        self.use_cosine_attention = True

        # Attention dropout
        self.attn_dropout = nn.Dropout(0.2)

    def forward(self, sequences):
        """
        _summary_

        :param sequences: _description_
        :return: _description_
        """
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                if getattr(self, 'use_cosine_attention', False):
                    q = F.normalize(q, p=2, dim=-1)
                    k = F.normalize(k, p=2, dim=-1)
                    scale = self.attn_temperature[head].clamp_min(0.01)  # avoid zero scale
                    attn_logits = (q @ k.T) * scale
                else:
                    attn_logits = q @ k.T / (self.d_head ** 0.5)

                attention = self.softmax(attn_logits)
                attention = self.attn_dropout(attention)

                seq_result.append(attention @ v)

            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


# ---------------------------------------------------- DropPath -------------------------------------------------------
#
#
class DropPath(nn.Module):
    """
    Implements Stochastic Depth (a.k.a. DropPath) regularization.

    During training, this module randomly drops entire residual paths (e.g., sublayers within a transformer block) with
    probability `drop_prob`, and scales the surviving paths accordingly. This encourages the model to learn more robust
    representations by reducing reliance on any individual layer.

    This is typically used in residual connections like:
        x = x + drop_path(f(x))

    Args:
        drop_prob (float): Probability of dropping the path (default: 0.1).

    Forward Args:
        x (Tensor): Input tensor of shape (B, ...), where B is the batch size.

    Returns:
        Tensor: Either the input unchanged (if not training or drop_prob=0), or the
                input scaled by 1 / (1 - drop_prob) and randomly zeroed per sample.
    """


    def __init__(self, drop_prob=0.1):
        """
        Initialize the DropPath module.

        Args:
            drop_prob (float): Probability of dropping the entire path. Must be in [0.0, 1.0].
        """
        super().__init__()
        self.drop_prob = drop_prob


    def forward(self, x):
        """
        Apply DropPath to the input tensor during training.

        Args:
            x (Tensor): Input tensor of shape (B, ...), where B is the batch size.

        Returns:
            Tensor: The output tensor with some residual paths dropped and rescaled during training.
                    If not in training mode or drop_prob is 0, returns input unchanged.
        """

        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor


# ------------------------------------------ Transformer Encoder Block ------------------------------------------------
#
# region Encoder
class TransformerEncoderBlock(nn.Module):
    """
    The transformer encoder consists of alternating layers of multi-headed self-attention and MLP blocks.
    Each block is composed of : A Layer Normalization (LN), followed by a Multi-head Self Attention (MHSA) and a
    residual connection. Then a second LN, a Multi-Layer Perceptron (MLP), and again a residual connection.

    A residual connection consists in just adding the original input to the result of some computation.
    This, intuitively, allows a network to become more powerful while also preserving the set of possible
    functions that the model can approximate.

    These blocks are connected back-to-back and referred to as "Layers" in the original paper.
    """
    def __init__(self, hidden_d, n_heads, mlp_ratio=4, debug_mode: bool = False):
        """
        _summary_

        :param hidden_d: _description_
        :param n_heads: _description_
        :param mlp_ratio: _description_, defaults to 4
        """
        super(TransformerEncoderBlock, self).__init__()

        self.hidden_d   = hidden_d
        self.n_heads    = n_heads
        self.debug_mode = debug_mode

        # Residual Connections — Will add the original input to the result of some computation.
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MHSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)

        # Call self.drop_path(...) around each sub-layer (mhsa and mlp) before adding it back to the residual.
        self.drop_path = DropPath(0.1)

        self.gamma_1 = nn.Parameter(torch.ones(hidden_d) * 1e-5)
        self.gamma_2 = nn.Parameter(torch.ones(hidden_d) * 1e-5)

        # The MLP is composed of two layers, where the hidden layer is four times as big (this is a parameter).
        mlp_dim = mlp_ratio * hidden_d
        transformer_dropout = 0.2

        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_dim),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(mlp_dim, hidden_d),
            nn.Dropout(transformer_dropout),
        )

    def forward(self, x):
        # 1) attention block with stochastic depth
        res = x
        # x = res + self.drop_path(self.mhsa(self.norm1(res)))
        x = res + self.drop_path(self.mhsa(self.norm1(res)) * self.gamma_1)

        # 2) MLP block with stochastic depth
        res = x
        # x = res + self.drop_path(self.mlp(self.norm2(res)))
        x = res + self.drop_path(self.mlp(self.norm2(res)) * self.gamma_2)

        return x


# ------------------------------------------- Main ViT Implementation -------------------------------------------------
#
# region ViT
class MmViT(Module):
    """
    Defines a Vision Transformer that uses the microbiome map images from community abundance profiles.
    """

    def __init__(self, chw=(3, 256, 256), n_patches:int = 8, hidden_d:int = 8, n_blocks:int = 2,
                 n_heads:int = 2, num_classes:int = 10, debug_mode:bool = True,
                 is_binary_class:bool = True, is_multi_label:bool = False):
        """
        Class constructor & initialization method.

        :param chw:                 Image Channel, Height, Width dimensions.
        :param n_patches:           Number of patches that we will find both in width and height.
        :param hidden_d:            Size of hidden dimension to map to.
        :param is_binary_class:     Are we doing binary classification?
        :param is_multi_label:      Multi-label classification?
        """

        super(MmViT, self).__init__()

        #   Main Attributes
        self.chw        = chw           # Input shape, as in Channels, Height, Width.
        self.n_patches  = n_patches     # Number of patches to split image into (in one dimension).
        self.hidden_d   = hidden_d      # Size of dimension that maps patch arrays to patch embedding vectors.
        self.n_blocks   = n_blocks      # Number of transformer blocks.
        self.n_heads    = n_heads       # Number of heads for encoder block.
        self.debug_mode = debug_mode    # Sets the debug mode on for verbose printing.
        self.is_binary_class = is_binary_class  # Will determine what the last layer in the MLP module is.
        self.is_multi_label = is_multi_label  # Multi-label classification?

        # Input and patches sizes.
        assert chw[1] % n_patches == 0, f"Input shape not divisible by number of patches: {chw[1]}"
        assert chw[2] % n_patches == 0, f"Input shape not divisible by number of patches: {chw[2]}"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # ---------------------------------- 1) Image Tokenization ------------------------------------------
        #
        # Process patches through a linear projection layer to get initial patch embeddings.

        # Input size of the images (includes channels).
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])

        self.layer_norm = nn.LayerNorm(normalized_shape=self.input_d)
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.pretoken = nn.Sequential(
            nn.Conv2d(chw[0], chw[0], kernel_size=3, padding=1, groups=chw[0], bias=False),
            nn.BatchNorm2d(chw[0]),
            nn.Conv2d(chw[0], chw[0], kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout2d(0.05)
        )

        # ---------------------------- 2) Learnable Classification Token ------------------------------------
        #
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # --------------------------------- 3) Positional Embedding -----------------------------------------
        #
        self.positional_embeddings = nn.Parameter(
            torch.zeros(n_patches**2 + 1, hidden_d)
        )
        nn.init.trunc_normal_(self.positional_embeddings, std=0.02)

        # ------------------------------- 4) Transformer Encoder Block --------------------------------------
        #
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # -------------------------------- 5) Classification MLP Head ---------------------------------------
        #
        bottleneck_dim = self.hidden_d
        dropout_rate  = 0.5

        modules = [
            # 1) Project down to a smaller “bottleneck” space
            nn.Linear(hidden_d, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # 2) Project up to the number of classes
            nn.Linear(bottleneck_dim // 2, num_classes)
        ]

        self.mlp = nn.Sequential(*modules)

    # region ViT - forward()
    def forward(self, images):
        """
        Forward pass of the model.

        :param images: _description_
        :return: _description_
        """

        images = self.pretoken(images)

        # Divide images into patches.
        n, c, height, width = images.shape

        patch_size = self.patch_size[1]
        total_n_patches = int((height * width) / (patch_size**2))

        # Image Patches.
        # Each patch is a Height/n_patches x Height/n_patches dimensional vector.
        # E.g., for a 256 image split into 8 patches with 3 channels, we'll have 256/8 = 32,
        #   or (32x32)x3 = 3072 sized dimensional vector.
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Image tokenization
        # Maps the vector corresponding to each patch to the hidden size dimension.
        tokens = self.layer_norm(patches)
        tokens = self.linear_mapper(tokens)

        # Add the classification token to the transformer tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add the positional embedding (patch tokens and positional embeddings).
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Encoder Blocks
        for block in self.encoder_blocks:
            out = block(out)

        # Extract just the classification token (first item) out of our N sequences, and
        # use each token to get N classifications.
        out = out[:, 0]

        logits = self.mlp(out)
        if self.is_multi_label:
            return torch.sigmoid(logits)
        else:
            return logits
