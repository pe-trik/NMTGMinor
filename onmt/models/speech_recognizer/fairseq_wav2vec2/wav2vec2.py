# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import compute_mask_indices, get_activation_fn, get_available_activation_fns
from .enum import ChoiceEnum
from torch.cuda.amp import autocast

from .fairseq_modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    MultiheadAttention,
    SamePad,
    TransposeLast,
    index_copy
)

from onmt.modules.layer_norm import LayerNorm
from onmt.modules.optimized.dropout_add import fused_dropout_add

from .utils import buffered_arange, index_put, is_xla_tensor

from fairseq.dataclass import FairseqDataclass
from fairseq.models.wav2vec import Wav2Vec2Config

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        if not module.fast_attention:
            normal_(module.q_proj.weight.data)
            normal_(module.k_proj.weight.data)
            normal_(module.v_proj.weight.data)
        else:
            normal_(module.proj_weight.data)


#
# @dataclass
# class Wav2Vec2Config(FairseqDataclass):
#     extractor_mode: EXTRACTOR_MODE_CHOICES = field(
#         default="default",
#         metadata={
#             "help": "mode for feature extractor. default has a single group norm with d "
#             "groups in the first conv block, whereas layer_norm has layer norms in "
#             "every block (meant to use with normalize=True)"
#         },
#     )
#     encoder_layers: int = field(
#         default=12, metadata={"help": "num encoder layers in the transformer"}
#     )
#     encoder_embed_dim: int = field(
#         default=768, metadata={"help": "encoder embedding dimension"}
#     )
#     encoder_ffn_embed_dim: int = field(
#         default=3072, metadata={"help": "encoder embedding dimension for FFN"}
#     )
#     encoder_attention_heads: int = field(
#         default=12, metadata={"help": "num encoder attention heads"}
#     )
#     activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
#         default="gelu", metadata={"help": "activation function to use"}
#     )
#
#     # dropouts
#     dropout: float = field(
#         default=0.1, metadata={"help": "dropout probability for the transformer"}
#     )
#     attention_dropout: float = field(
#         default=0.1, metadata={"help": "dropout probability for attention weights"}
#     )
#     activation_dropout: float = field(
#         default=0.0, metadata={"help": "dropout probability after activation in FFN"}
#     )
#     encoder_layerdrop: float = field(
#         default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
#     )
#     dropout_input: float = field(
#         default=0.0,
#         metadata={"help": "dropout to apply to the input (after feat extr)"},
#     )
#     dropout_features: float = field(
#         default=0.0,
#         metadata={"help": "dropout to apply to the features (after feat extr)"},
#     )
#
#     final_dim: int = field(
#         default=0,
#         metadata={
#             "help": "project final representations and targets to this many dimensions."
#             "set to encoder_embed_dim is <= 0"
#         },
#     )
#     layer_norm_first: bool = field(
#         default=False, metadata={"help": "apply layernorm first in the transformer"}
#     )
#     conv_feature_layers: str = field(
#         default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
#         metadata={
#             "help": "string describing convolutional feature extraction layers in form of a python list that contains "
#             "[(dim, kernel_size, stride), ...]"
#         },
#     )
#     conv_bias: bool = field(
#         default=False, metadata={"help": "include bias in conv encoder"}
#     )
#     logit_temp: float = field(
#         default=0.1, metadata={"help": "temperature to divide logits by"}
#     )
#     quantize_targets: bool = field(
#         default=False, metadata={"help": "use quantized targets"}
#     )
#     quantize_input: bool = field(
#         default=False, metadata={"help": "use quantized inputs"}
#     )
#     same_quantizer: bool = field(
#         default=False, metadata={"help": "use same quantizer for inputs and targets"}
#     )
#     target_glu: bool = field(
#         default=False, metadata={"help": "adds projection + glu to targets"}
#     )
#     feature_grad_mult: float = field(
#         default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
#     )
#     quantizer_depth: int = field(
#         default=1,
#         metadata={"help": "number of quantizer layers"},
#     )
#     quantizer_factor: int = field(
#         default=3,
#         metadata={
#             "help": "dimensionality increase for inner quantizer layers (if depth > 1)"   
#         },
#     )
#     latent_vars: int = field(
#         default=320,
#         metadata={"help": "number of latent variables V in each group of the codebook"},
#     )
#     latent_groups: int = field(
#         default=2,
#         metadata={"help": "number of groups G of latent variables in the codebook"},
#     )
#     latent_dim: int = field(
#         default=0,
#         metadata={
#             "help": "if > 0, uses this dimensionality for latent variables. "
#             "otherwise uses final_dim / latent_groups"
#         },
#     )
#
#     # masking
#     mask_length: int = field(default=10, metadata={"help": "mask length"})
#     mask_prob: float = field(
#         default=0.65, metadata={"help": "probability of replacing a token with mask"}
#     )
#     mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
#         default="static", metadata={"help": "how to choose mask length"}
#     )
#     mask_other: float = field(
#         default=0,
#         metadata={
#             "help": "secondary mask argument (used for more complex distributions), "
#             "see help in compute_mask_indices"
#         },
#     )
#     no_mask_overlap: bool = field(
#         default=False, metadata={"help": "whether to allow masks to overlap"}
#     )
#     mask_min_space: int = field(
#         default=1,
#         metadata={"help": "min space between spans (if no overlap is enabled)"},
#     )
#
#     # channel masking
#     mask_channel_length: int = field(
#         default=10, metadata={"help": "length of the mask for features (channels)"}
#     )
#     mask_channel_prob: float = field(
#         default=0.0, metadata={"help": "probability of replacing a feature with 0"}
#     )
#     mask_channel_before: bool = False
#     mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
#         default="static",
#         metadata={"help": "how to choose mask length for channel masking"},
#     )
#     mask_channel_other: float = field(
#         default=0,
#         metadata={
#             "help": "secondary mask argument (used for more complex distributions), "
#             "see help in compute_mask_indicesh"
#         },
#     )
#     no_mask_channel_overlap: bool = field(
#         default=False, metadata={"help": "whether to allow channel masks to overlap"}
#     )
#     mask_channel_min_space: int = field(
#         default=1,
#         metadata={"help": "min space between spans (if no overlap is enabled)"},
#     )
#
#     # negative selection
#     num_negatives: int = field(
#         default=100,
#         metadata={"help": "number of negative examples from the same sample"},
#     )
#     negatives_from_everywhere: bool = field(
#         default=False,
#         metadata={"help": "sample negatives from everywhere, not just masked states"},
#     )
#     cross_sample_negatives: int = field(
#         default=0, metadata={"help": "number of negative examples from the any sample"}
#     )
#     codebook_negatives: int = field(
#         default=0, metadata={"help": "number of negative examples codebook"}
#     )
#
#     # positional embeddings
#     conv_pos: int = field(
#         default=128,
#         metadata={"help": "number of filters for convolutional positional embeddings"},
#     )
#     conv_pos_groups: int = field(
#         default=16,
#         metadata={"help": "number of groups for convolutional positional embedding"},
#     )
#
#     latent_temp: Tuple[float, float, float] = field(
#         default=(2, 0.5, 0.999995),
#         metadata={
#             "help": "temperature for latent variable sampling. "
#             "can be tuple of 3 values (start, end, decay)"
#         },
#     )


# @register_model("wav2vec2", dataclass=Wav2Vec2Config)
class Wav2Vec2Model(torch.nn.Module):
    def __init__(self, cfg: Wav2Vec2Config,
                 favor=False, feature_redraw_interval=1000, auto_check_redraw=True,
                 weight_drop=0.0):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before if hasattr(cfg, 'mask_channel_before') else True
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth if hasattr(cfg, 'quantizer_depth') else 1,
                weight_proj_factor=cfg.quantizer_factor if hasattr(cfg, 'quantizer_factor') else 3,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                    weight_proj_depth=cfg.quantizer_depth,
                    weight_proj_factor=cfg.quantizer_factor,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg, favor=favor, weight_drop=weight_drop)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
        self.favor = favor

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def clean_unused_weights(self):

        self.input_quantizer = None
        self.quantizer = None
        self.target_glu = None
        self.final_proj = None
        self.project_q = None

        return

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(
            self,
            x,
            padding_mask,
            mask_indices=None,
            mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb.type_as(x))
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                        .to(x.device)
                        .unsqueeze(1)
                        .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, self.n_negatives)
                        .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, self.cross_sample_negatives)
                        .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits = logits / self.logit_temp

        if is_xla_tensor(logits) or neg_is_pos.any():
            fillval = -float(2 ** 30)
            if not hasattr(self, "_inftensor"):
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
            self,
            source,
            padding_mask=None,
            mask=True,
            features_only=False,
            layer=None,
            mask_indices=None,
            mask_channel_indices=None,
            padding_count=None,
            precomputed_tdnn=False,
            quantize=False, quantize_only=False,
            lang=None,
            mixture=None
    ):
        # if the tdnn features are precomputed then skip them
        if not precomputed_tdnn:
            if self.feature_grad_mult > 0 or source.requires_grad:
                features = self.feature_extractor(source)
                if self.feature_grad_mult != 1.0:
                    features = GradMultiply.apply(features, self.feature_grad_mult)
            else:
                with torch.no_grad():
                    features = self.feature_extractor(source)

            if not features_only:
                features_pen = features.float().pow(2).mean()

            # transpose from B x C x T to B x T x C (because conv takes input as B x 1 x T)
            features = features.transpose(1, 2)
        else:
            features = source

        # perform layer norm ... but check grad mode
        current_grad_mode = torch.is_grad_enabled()
        if current_grad_mode:
            torch.set_grad_enabled(self.layer_norm.weight.requires_grad)
        features = self.layer_norm(features)
        torch.set_grad_enabled(current_grad_mode)

        if quantize:
            assert self.quantizer is not None
            with torch.no_grad():
                quantizer_output = self.quantizer.forward_idx(features)
        else:
            quantizer_output = None

        if features_only:
            unmasked_features = None
        else:
            unmasked_features = features.clone()

        if not precomputed_tdnn:  # then compute the padding mask after the TDNN step
            if padding_mask is not None and padding_mask.any():
                input_lengths = (1 - padding_mask.long()).sum(-1)
                # apply conv formula to get real output_lengths
                output_lengths = self._get_feat_extract_output_lengths(input_lengths)

                padding_mask = torch.zeros(
                    features.shape[:2], dtype=features.dtype, device=features.device
                )

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
            else:
                padding_mask = None

        if quantize_only:
            quantized_x, quantized_target = quantizer_output
            output_dict = dict()
            output_dict['quantized_x'] = quantized_x  # b x t x ?
            output_dict['quantized_target'] = quantized_target  # b x t x num_groups
            output_dict['padding_mask'] = padding_mask
            return output_dict

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        # unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        # if self.input_quantizer:
        #     q = self.input_quantizer(features, produce_targets=False)
        #     features = q["x"]
        #     num_vars = q["num_vars"]
        #     code_ppl = q["code_perplexity"]
        #     prob_ppl = q["prob_perplexity"]
        #     curr_temp = q["temp"]
        #     features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None and not features_only:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer, lang=lang, mixture=mixture)

        if features_only:
            output_dict =  {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

            if quantize:
                quantized_x, quantized_target = quantizer_output
                output_dict['quantized_x'] = quantized_x  # b x t x ?
                output_dict['quantized_target'] = quantized_target # b x t x num_groups
                # print(quantized_x.size(), quantized_target.size())
                # print(quantized_target)

            return output_dict

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands = self.quantizer(unmasked_features, produce_targets=False)[
                    "x"
                ]
                negs, _ = self.sample_negatives(
                    neg_cands,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_conv_features(self, source, padding_mask):

        with torch.no_grad():
            features = self.feature_extractor(source)

        # transpose from B x C x T to B x T x C (because conv takes input as B x 1 x T)
        features = features.transpose(1, 2).contiguous()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            bsz, seq_len = features.size(0), features.size(1)
            padding_mask = features.new(bsz, seq_len).zero_()

        return features, padding_mask.long()

    def extract_features(self, source, padding_mask, mask=False, layer=None, precomputed_tdnn=False,
                         lang=None, mixture=None):
        res = self.forward(
            source, padding_mask, mask=mask, features_only=True, layer=layer, precomputed_tdnn=precomputed_tdnn,
            lang=lang, mixture=mixture
        )
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self, removing_quantizer=True):
        if removing_quantizer:
            self.quantizer = None
        else:
            print("[INFO] Keeping the quantizer")
            print(self.quantizer)
            # self.groups = groups
            # self.combine_groups = combine_groups
            # self.input_dim = dim
            # self.num_vars = num_vars
            # self.time_first = time_first
            print("Groups: ", self.quantizer.groups)
            print("Combine groups: ", self.quantizer.combine_groups)
            print("num vars: ", self.quantizer.num_vars)
            print(self.quantizer.vars.size())
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

    def convert_fast_attention(self):

        model = self.encoder

        def find_modules(nn_module, type):
            return [module for module in nn_module.modules() if isinstance(module, type)]

        fast_attentions = find_modules(model, MultiheadAttention)
        for fast_attention in fast_attentions:
            fast_attention.convert_fast_attention()

    # Convert the self-attn module to deep speed transformer
    def convert_deepspeed(self, training=True, bsz=32):
        self.encoder.convert_deepspeed(training=training, bsz=bsz)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT (only for waveforms with 1 channel)
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args, favor=False, weight_drop=0.0):
        """
        :param args:
        :param favor: Performer Attention
        """
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.favor = favor
        self.weight_drop = weight_drop
        self.num_heads = args.encoder_attention_heads
        self.num_layers = args.encoder_layers
        self.attention_dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.deepspeed = False

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    weight_drop=self.weight_drop,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    favor=favor
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

        from onmt.modules.optimized.fast_mha import fast_bert_mha
        self.fast_bert_mha = fast_bert_mha

    def forward(self, x, padding_mask=None, layer=None, lang=None, mixture=None):
        x, layer_results = self.extract_features(x, padding_mask, layer, lang=lang, mixture=mixture)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None, lang=None, mixture=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        can_run_fast_bert_mha = False
        # check if fast bert mha can be run
        seq_len = x.size(1)
        bsz = x.size(0)
        sm = torch.cuda.get_device_capability()
        total_bsz = 0

        if self.deepspeed:
            fast_attention = False
        else:
            fast_attention = self.layers[0].self_attn.fast_attention

        # only run this when seq_len <= 512 and sm = 80/86 and type = half
        if self.fast_bert_mha and (seq_len <= 512 and bsz >= 4 and sm[0] == 8 and sm[1] == 0) \
                and not self.deepspeed and fast_attention and x.dtype == torch.half:
            print("[INFO] Can run FAST MHA with seq_len", seq_len)
            can_run_fast_bert_mha = True

            # masked positions = 1 so to compute length we need the (1 -)
            if padding_mask is None:
                padding_mask = x.new_zeros(bsz, seq_len)
            padding_mask = padding_mask.long()
            lengths = (1 - padding_mask).sum(dim=1)
            lengths = lengths.cpu().tolist()  # list of lengths for B seqs

            # remove paddings from x
            x = x.view(-1, x.size(-1))
            non_pad_indices = torch.nonzero(padding_mask.view(-1).ne(1)).squeeze(1)
            x = x.index_select(0, non_pad_indices)

            # maybe pad it so the first dim % 8 = 0?
            total_bsz = x.size(0)
            # if total_bsz % 16 != 0:
            #     patch_size = total_bsz - (total_bsz // 16) * 16
            #     patch = torch.randn(patch_size, x.size(-1), dtype=x.dtype, device=x.device)
            #     x = torch.cat([x, patch], dim=0)
            #     # treat the "tail" like a normal sequence that attends to itself
            #     lengths.append(patch_size)

            max_len = max(lengths)
            # cumulative sequence lengths (required input for fmha)
            a = torch.tensor(np.array([0] + lengths), dtype=torch.int32)
            cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=x.device)
        else:
            # print("[INFO] CanNOT run FAST MHA with seq_len", seq_len)
            max_len = -1
            cu_seqlens = None
            non_pad_indices = None

        if not self.favor and not self.deepspeed and not can_run_fast_bert_mha:
            # B x T x C -> T x B x C  (only for vanilla self-attention)
            x = x.transpose(0, 1)
        x = x.contiguous()

        # If not using deepspeed: proceed like normal
        if not self.deepspeed:
            layer_results = []
            r = None
            for i, layer in enumerate(self.layers):
                dropout_probability = np.random.random()
                if not self.training or (dropout_probability > self.layerdrop):
                    x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False,
                                 max_len=max_len, cu_seqlens=cu_seqlens,
                                 lang=lang, mixture=mixture)
                    if tgt_layer is not None:
                        layer_results.append((x, z))
                if i == tgt_layer:
                    r = x
                    break

            if r is not None:
                x = r

            # T x B x C -> B x T x C
            if not self.favor and not can_run_fast_bert_mha:
                x = x.transpose(0, 1)

        else:
            # deepspeed has strict requirement so better disable autocast
            with autocast(enabled=False):
                dtype = x.dtype
                x = x.half()

                layer_results = []
                if padding_mask is None:
                    padding_mask = x.new(x.size(0), x.size(1)).fill_(0)
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
                padding_mask = padding_mask.type_as(x).fill_(-10000)
                for i, layer in enumerate(self.layers):
                    dropout_probability = np.random.random()
                    if not self.training or (dropout_probability > self.layerdrop):
                        x = layer(x, padding_mask)

            x = x.to(dtype)

        # if we remove padding before (for fast bert MHA) then remember to put padding back
        # to restore the form B x T X H
        if can_run_fast_bert_mha:
            # remove the patch
            if x.size(0) > total_bsz:
                x = x[:total_bsz, :]
            x = index_copy(x, non_pad_indices, bsz * seq_len)
            x = x.view(bsz, seq_len, -1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def add_adapters(self, n_languages, adapter_location=1):

        for layer in self.layers:
            layer.add_adapters(n_languages, adapter_location=adapter_location)

    def add_factorize(self, n_languages, rank=4, multiplicative=False, fast=False):

        for layer in self.layers:
            layer.add_factorized(n_languages, rank=rank, multiplicative=multiplicative, fast=fast)

    def freeze_or_unfreeze_ffn_params(self):

        for layer in self.layers:
            for p in layer.fc1.parameters():
                p.requires_grad = not p.requires_grad
            for p in layer.fc2.parameters():
                p.requires_grad = not p.requires_grad

    def convert_deepspeed(self, training=True, bsz=32):

        from deepspeed import DeepSpeedTransformerConfig, DeepSpeedTransformerLayer
        self.deepspeed = True

        config = DeepSpeedTransformerConfig(batch_size=bsz,
                                            hidden_size=self.embedding_dim,
                                            heads=self.num_heads,
                                            attn_dropout_ratio=self.attention_dropout,
                                            hidden_dropout_ratio=self.activation_dropout,
                                            num_hidden_layers=-1,
                                            initializer_range=0.02,
                                            local_rank=-1,
                                            seed=1234,
                                            fp16=True,
                                            pre_layer_norm=True,
                                            adjust_init_range=False,
                                            attn_dropout_checkpoint=False,
                                            normalize_invertible=False,
                                            gelu_checkpoint=False,
                                            return_tuple=False,
                                            training=training)

        old_layers = self.layers

        self.layers = nn.ModuleList([
            copy.deepcopy(DeepSpeedTransformerLayer(config))
            for _ in range(self.num_layers)
        ])

        # self.layers = nn.ModuleList()
        # for i in range(self.num_layers):
        #
        #     old_layer = old_layers[i]
        #
        # new_layer = DeepSpeedTransformerLayer(config)
        #
        #     with torch.no_grad():
        #         new_layer.attn_qkvw.copy_(old_layer.self_attn.proj_weight)
        #         new_layer.attn_qkvb.copy_(old_layer.self_attn.proj_bias)
        #         new_layer.attn_ow.copy_(old_layer.self_attn.out_proj.weight)
        #         new_layer.attn_ob.copy_(old_layer.self_attn.out_proj.bias)
        #         new_layer.attn_nw.copy_(old_layer.self_attn_layer_norm.weight)
        #         new_layer.attn_nb.copy_(old_layer.self_attn_layer_norm.bias)
        #         new_layer.inter_w.copy_(old_layer.fc1.weight)
        #         new_layer.inter_b.copy_(old_layer.fc1.bias)
        #         new_layer.output_w.copy_(old_layer.fc2.weight)
        #         new_layer.output_b.copy_(old_layer.fc2.bias)
        #         new_layer.norm_w.copy_(old_layer.final_layer_norm.weight)
        #         new_layer.norm_b.copy_(old_layer.final_layer_norm.bias)
        #
        # self.layers.append(new_layer)


# noinspection PyAttributeOutsideInit
class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            weight_drop: float = 0.0,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
            favor=False
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.favor = favor
        self.has_adapter = False
        self.is_factorized = False
        self.fast_factorize = False
        self.multiplicative_factorize = False

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.activation_fn_name = activation_fn
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            weight_drop=weight_drop,
            self_attention=True,
            favor=favor
        )

        self.residual_dropout = dropout
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(self.activation_dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=False)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.fused = False
        self.fused_function = None
        if self.activation_fn_name == 'relu':
            from onmt.modules.mlp.mlp import mlp_relu_function
            if mlp_relu_function is not None:
                self.fused_function = mlp_relu_function
                self.fused = True
        elif self.activation_fn_name == 'gelu':
            from onmt.modules.mlp.mlp import mlp_gelu_function

            if mlp_gelu_function is not None:
                self.fused_function = mlp_gelu_function
                self.fused = True

    def add_adapters(self, n_languages, downsampling_factor=4, adapter_location=1):
        """
        :param n_languages: one adapter per language
        :param downsampling_factor: downsampling rate size for the hidden layer
        :param adapter_location:
        :return:
        """

        self.n_languages = n_languages
        self.has_adapter = True
        self.adapter_location = adapter_location
        from .adapter import MultilingualAdapter
        self.adapter = MultilingualAdapter(n_languages, self.embedding_dim, downsample_factor=downsampling_factor)

        if adapter_location == 2:
            self.mid_adapter = MultilingualAdapter(n_languages, self.embedding_dim,
                                                   downsample_factor=downsampling_factor)

    def add_factorized(self, n_languages, rank=4, multiplicative=True, fast=False):

        self.self_attn.add_factorized_weights(n_languages, rank=rank, multiplicative=multiplicative)
        self.multiplicative_factorize = multiplicative
        self.is_factorized = True
        self.fast_factorize = fast

        embed_dim = self.embedding_dim
        ffn_dim = self.ffn_embedding_dim
        self.r_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_embedding_dim))
        self.s_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embedding_dim))
        self.r_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embedding_dim))
        self.s_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_embedding_dim))

        nn.init.normal_(self.r_i, 0.0, 0.02)
        nn.init.normal_(self.s_i, 0.0, 0.02)
        nn.init.normal_(self.r_o, 0.0, 0.02)
        nn.init.normal_(self.s_o, 0.0, 0.02)

        if multiplicative:
            rank = rank if fast else 1
            self.rm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_embedding_dim))
            self.sm_i = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embedding_dim))
            self.rm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.embedding_dim))
            self.sm_o = torch.nn.Parameter(torch.Tensor(n_languages, rank, self.ffn_embedding_dim))

            constant = math.sqrt(1.0 / rank) if fast else 1
            nn.init.constant_(self.rm_i, constant)
            nn.init.constant_(self.sm_i, constant)
            nn.init.constant_(self.rm_o, constant)
            nn.init.constant_(self.sm_o, constant)

    def get_mlp_weights(self, lang=None, mixture=None):

        in_weight = self.fc1.weight
        out_weight = self.fc2.weight
        in_bias = self.fc1.bias
        out_bias = self.fc2.bias

        if lang is not None:
            assert mixture is None

            if self.is_factorized:
                if self.multiplicative_factorize:
                    rm_i = torch.index_select(self.rm_i, 0, lang).squeeze(0)  # squeeze possible because only 1
                    sm_i = torch.index_select(self.sm_i, 0, lang).squeeze(0)
                    rm_o = torch.index_select(self.rm_o, 0, lang).squeeze(0)
                    sm_o = torch.index_select(self.sm_o, 0, lang).squeeze(0)

                    if self.fast_factorize:
                        mul_factor_in = torch.mm(rm_i.t(), sm_i)
                        mul_factor_out = torch.mm(rm_o.t(), sm_o)
                    else:
                        mul_factor_in = torch.bmm(rm_i.unsqueeze(-1), sm_i.unsqueeze(1)).sum(dim=0)
                        mul_factor_out = torch.bmm(rm_o.unsqueeze(-1), sm_o.unsqueeze(1)).sum(dim=0)

                    in_weight = in_weight * mul_factor_in
                    out_weight = out_weight * mul_factor_out

                r_i = torch.index_select(self.r_i, 0, lang).squeeze(0)
                s_i = torch.index_select(self.s_i, 0, lang).squeeze(0)
                r_o = torch.index_select(self.r_o, 0, lang).squeeze(0)
                s_o = torch.index_select(self.s_o, 0, lang).squeeze(0)

                if self.fast_factorize:
                    add_factor_in = torch.mm(r_i.t(), s_i)
                    add_factor_out = torch.mm(r_o.t(), s_o)
                else:
                    add_factor_in = torch.bmm(r_i.unsqueeze(-1), s_i.unsqueeze(1)).sum(dim=0)
                    add_factor_out = torch.bmm(r_o.unsqueeze(-1), s_o.unsqueeze(1)).sum(dim=0)

                in_weight = in_weight + add_factor_in
                out_weight = out_weight + add_factor_out

        if mixture is not None:
            raise NotImplementedError

        return in_weight, out_weight, in_bias, out_bias

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            max_len=-1, cu_seqlens=None,
            lang=None, mixture=None,
            **kwargs
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        is_fast = self.self_attn.fast_attention

        def call_mlp(x, in_weight, out_weight, in_bias, out_bias, activation_fn, dropout_p, training_,
                     fused, fused_function):

            # TODO: check type x torch.half or torch.float32
            if fused and x.is_cuda and is_fast:
                dropout_p_ = dropout_p if training_ else 0.0

                weights = [in_weight, out_weight]
                biases = [in_bias, out_bias]

                x = fused_function(dropout_p_, False, x, *weights, *biases)

            else:
                x = F.linear(x, in_weight, in_bias)
                x = activation_fn(x)
                x = F.dropout(x, dropout_p, training=training_)
                x = F.linear(x, out_weight, out_bias)

            return x

        if self.has_adapter:
            if self.adapter_location == 1:
                assert lang is not None or mixture is not None
                x = self.adapter(x, lang=lang, mixture=mixture)

            x.add_(residual)  # residual is before the big FFN
            residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x, fast=is_fast)

            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                max_len=max_len, cu_seqlens=cu_seqlens,
                lang=lang, mixture=mixture
            )

            if is_fast:
                x = fused_dropout_add(x, residual, self.residual_dropout, self.training)
            else:
                x = self.dropout1(x) + residual

            residual = x

            if self.has_adapter:
                if self.adapter_location == 2:
                    assert lang is not None or mixture is not None
                    x = self.mid_adapter(x, lang=lang, mixture=mixture)
                    x.add_(residual)
                    residual = x

            x = self.final_layer_norm(x, fast=is_fast)

            in_weight, out_weight, in_bias, out_bias = self.get_mlp_weights(lang=lang, mixture=mixture)
            x = call_mlp(x, in_weight, out_weight, in_bias, out_bias, self.activation_fn,
                         self.dropout2.p, self.training,
                         self.fused, self.fused_function)

            if is_fast:
                x = fused_dropout_add(x, residual, self.residual_dropout, self.training)
            else:
                x = self.dropout3(x)
                x = residual + x

        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x

            if self.fused and x.is_cuda:
                dropout = self.dropout2.p if self.training else 0.0
                if self.fused_blaslt and dropout == 0.0:
                    x = self.fused_function(x.view(seq_len * bsz, -1), self.fc1.weight, self.fc1.bias,
                                            self.fc2.weight, self.fc2.bias)
                else:
                    weights = [self.fc1.weight, self.fc2.weight]
                    biases = [self.fc1.bias, self.fc2.bias]

                    seq_len, bsz, hidden_size = x.size(0), x.size(1), x.size(2)

                    x = self.fused_function(dropout, False, x.view(seq_len * bsz, -1),
                                            *weights, *biases)

                x = x.view(seq_len, bsz, hidden_size)
            else:
                x = self.activation_fn(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn