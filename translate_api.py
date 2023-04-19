#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import logging

import onmt.markdown
import torch
import argparse
import math
import numpy as np
from onmt.inference.fast_translator import FastTranslator

def add_parser_args(parser):
    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-sub_model', required=False, default="",
                        help='Path to (secondary) model .pt file')
    parser.add_argument('-pretrained_classifier', required=False, default="",
                        help='Path to external classifier model .pt file')
    parser.add_argument('-streaming', action="store_true",
                        help="""Use streaming mode (for model with streaming)""")
    parser.add_argument('-lm', required=False,
                        help='Path to language model .pt file. Used for cold fusion')
    parser.add_argument('-vocab_list', default="",
                        help='A Vocabulary list (1 word per line). Only are these words generated during translation.')
    parser.add_argument('-vocab_id_list', default="",
                        help='A Vocabulary list (1 word per line). Only are these words generated during translation.')
    parser.add_argument('-autoencoder', required=False,
                        help='Path to autoencoder .pt file')
    parser.add_argument('-input_type', default="word",
                        help="Input type: word/char")
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-sub_src', required=False, default="",
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-past_src', required=False, default="",
                        help='Past Source sequence to decode (one line per sequence)')
    parser.add_argument('-src_lang', default='src',
                        help='Source language')
    parser.add_argument('-src_atb', default='nothingness',
                        help='Source language')
    parser.add_argument('-tgt_lang', default='tgt',
                        help='Target language')
    parser.add_argument('-tgt_atb', default='nothingness',
                        help='Target language')
    parser.add_argument('-attributes', default="",
                        help='Attributes for the decoder. Split them by | ')
    parser.add_argument('-ensemble_weight', default="",
                        help='Weight for ensembles. Default as uniform. Split them by | and they will be normalized later')
    parser.add_argument('-sub_ensemble_weight', default="",
                        help='Weight for ensembles. Default as uniform. Split them by | and they will be normalized later')

    parser.add_argument('-stride', type=int, default=1,
                        help="Stride on input features")
    parser.add_argument('-concat', type=str, default="1",
                        help="Concate sequential audio features to decrease sequence length")
    parser.add_argument('-asr_format', default="h5", required=False,
                        help="Format of asr data h5 or scp")
    parser.add_argument('-encoder_type', default='text',
                        help="Type of encoder to use. Options are [text|img|audio].")
    parser.add_argument('-previous_context', type=int, default=0,
                        help="Number of previous sentence for context")
    parser.add_argument('-max_memory_size', type=int, default=512,
                        help="Number of memory states stored in the buffer for XL models")

    parser.add_argument('-tgt',
                        help='True target sequence (optional)')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-prefix_string', default='',
                        help="""Prefix string for all of the translation""")
    parser.add_argument('-prefix_tgt', default='',
                        help="""Prefix file that contains prefix string for each of the translation
                        (must use either this or prefix_string, not both""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-max_sent_length', type=int, default=256,
                        help='Maximum sentence length.')
    parser.add_argument('-replace_unk', action="store_true",
                        help="""Replace the generated UNK tokens with the source
                        token that had highest attention weight. If phrase_table
                        is provided, it will lookup the identified source token and
                        give the corresponding target token. If it is not provided
                        (or the identified source token does not exist in the
                        table) then it will copy the source token""")
    parser.add_argument('-start_with_bos', action="store_true",
                        help="""Add BOS token to the top of the source sentence""")
    # parser.add_argument('-phrase_table',
    #                     help="""Path to source-target dictionary to replace UNK
    #                     tokens. See README.md for the format of this file.""")
    parser.add_argument('-verbose', action="store_true",
                        help='Print scores and predictions for each sentence')
    parser.add_argument('-sampling', action="store_true",
                        help='Using multinomial sampling instead of beam search')
    parser.add_argument('-dump_beam', type=str, default="",
                        help='File to dump beam information to.')
    parser.add_argument('-bos_token', type=str, default="<s>",
                        help='BOS Token (used in multilingual model). Default is <s>.')
    parser.add_argument('-no_bos_gold', action="store_true",
                        help='BOS Token (used in multilingual model). Default is <s>.')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_repeat_ngram_size', type=int, default=0,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-alpha', type=float, default=0.6,
                        help="""Length Penalty coefficient""")
    parser.add_argument('-beta', type=float, default=0.0,
                        help="""Coverage penalty coefficient""")
    parser.add_argument('-print_nbest', action='store_true',
                        help='Output the n-best list instead of a single sentence')
    parser.add_argument('-ensemble_op', default='mean', help="""Ensembling operator""")
    parser.add_argument('-normalize', action='store_true',
                        help='To normalize the scores based on output length')
    parser.add_argument('-no_buffering', action='store_true',
                        help='To remove buffering for transformer models (slower but more memory)')
    parser.add_argument('-src_align_right', action='store_true',
                        help='To normalize the scores based on output length')
    parser.add_argument('-fp16', action='store_true',
                        help='To use floating point 16 in decoding')
    parser.add_argument('-dynamic_quantile', type=int, default=0,
                        help='To use int8 in decoding (for linear and LSTM layers only).')
    parser.add_argument('-gpu', type=int, default=-1,
                        help="Device to run on")
    parser.add_argument('-fast_translate', action='store_true',
                        help='Using the fast decoder')
    parser.add_argument('-global_search', action='store_true',
                        help='Using the global beam search for streaming')
    parser.add_argument('-dynamic_max_len', action='store_true',
                        help='Using the fast decoder')
    parser.add_argument('-dynamic_max_len_scale', type=float, default=5.0,
                        help='Using the fast decoder')
    parser.add_argument('-dynamic_min_len_scale', type=float, default=0.0,
                        help='Using the fast decoder')
    parser.add_argument('-external_tokenizer', default="",
                        help="External tokenizer from Huggingface. Currently supports barts.")




def addone(f):
    for line in f:
        yield line
    yield None


def len_penalty(s, l, alpha):
    l_term = math.pow(l, alpha)
    return s / l_term


def get_sentence_from_tokens(tokens, ids, input_type, external_tokenizer=None):
    if external_tokenizer is None:
        if input_type == 'word':
            sent = " ".join(tokens)
        elif input_type == 'char':
            sent = "".join(tokens)
        else:
            raise NotImplementedError

    else:
        sent = external_tokenizer.decode(ids, True, True).strip()

    return sent


class TranlateAPI():

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens
        if len(tokens) > 0 and isinstance(tokens[0], (np.ndarray, list)):
            return [self.external_tokenizer.decode(ts, True, True).strip() for ts in tokens]
        if len(tokens) > 0 and isinstance(tokens[0], (torch.Tensor)):
            return self.external_tokenizer.decode(tokens[0].cpu().numpy(), True, True).strip()
        return self.external_tokenizer.decode(tokens, True, True).strip()

    def __init__(self, opt, beam_search_filtering=False) -> None:
        opt.cuda = opt.gpu > -1
        logging.info(f'Using GPU: {opt.cuda}')
        if opt.cuda:
            try:
                torch.cuda.set_device(opt.gpu)
            except:
                pass
        

        # Always pick n_best
        opt.n_best = opt.beam_size
        self.opt = opt

        translator = FastTranslator(opt)
        for m in translator.models:
            m.eval()
        self.BOS = translator.tgt_bos
        self.EOS = translator.tgt_eos
        logging.info(f"Using BOS: {self.BOS}")
        logging.info(f"Using EOS: {self.EOS}")

        if hasattr(translator, 'tgt_external_tokenizer'):
            external_tokenizer = translator.tgt_external_tokenizer
        else:
            external_tokenizer = None

        self.translator = translator
        self.external_tokenizer = external_tokenizer
        self.beam_search_filtering = beam_search_filtering

    def dict(self):
        return self.translator.tgt_dict.idxToLabel
        
    def infer(self, src, prefix, itype='wav', ctc_policy_hypothesis=None,beam_search_filtering=False):
        opt = self.opt
        translator = self.translator

        src_batches = []
        src_batch, tgt_batch, past_src_batch = [], [], []

        count = 0


        # Audio processing for the source batch
        if itype == 'fbank':

            """
            For Audio we will have to group samples by the total number of frames in the source
            """

            past_audio_data = open(opt.past_src) if opt.past_src else None
            past_src_batches = list()
            s_prev_context = []
            t_prev_context = []

            i = 0

            concats = opt.concat.split("|")

            n_models = len(opt.model.split("|"))
            if len(concats) == 1:
                concats = concats * n_models

            assert len(concats) == n_models, "The number of models must match the number of concat configs"
            for j, _ in enumerate(concats):
                src_batches.append(list())  # We assign different inputs for each model in the ensemble
                if past_audio_data:
                    past_src_batches.append(list())

            sub_src = open(opt.sub_src) if opt.sub_src else None
            sub_src_batch = list()

            try:
                line = src

            except StopIteration:
                pass

            if opt.stride != 1:
                line = line[0::opt.stride]
                if past_line: past_line = past_line[0::opt.stride]

            line = torch.from_numpy(line)
            past_line = torch.from_numpy(past_line) if past_audio_data else None

            original_line = line
            src_length = line.size(0)

            # handling different concatenation settings (for example 4|1|4)
            for j, concat_ in enumerate(concats):
                concat = int(concat_)
                line = original_line

                # TODO: move this block to function
                if concat != 1:
                    add = (concat - line.size()[0] % concat) % concat
                    z = torch.FloatTensor(add, line.size()[1]).zero_()
                    line = torch.cat((line, z), 0)
                    line = line.reshape((line.size()[0] // concat, line.size()[1] * concat))
                    if past_audio_data:
                        add = (concat - past_line.size()[0] % concat) % concat
                        z = torch.FloatTensor(add, past_line.size()[1]).zero_()
                        past_line = torch.cat((past_line, z), 0)
                        past_line = past_line.reshape((past_line.size()[0] // concat, past_line.size()[1] * concat))

                src_batches[j].append(line)
                if past_audio_data: past_src_batches[j].append(past_line)


            pred_batch, pred_ids, pred_score, pred_length, \
            gold_score, num_gold_words, all_gold_scores = translator.translate(
                src_batches,
                tgt_batch,
                past_src_data=past_src_batches,
                sub_src_data=sub_src_batch,
                type='asr', prefix=prefix,ctc_policy_hypothesis=ctc_policy_hypothesis)
            return pred_ids

        elif itype == 'wav':
            n_models = len(opt.model.split("|"))

            for j in range(n_models):
                src_batches.append(list())

            line = src

            if opt.stride != 1:
                line = line[0::opt.stride]

            original_line = line

            # handling different concatenation settings (for example 4|1|4)
            for j in range(n_models):
                src_batches[j].append(line)

            pred_batch, pred_ids, pred_score, pred_length, \
            gold_score, num_gold_words, all_gold_scores, ctc_hypothesis = translator.translate(
                src_batches,
                tgt_batch,
                past_src_data=[],
                sub_src_data=[], type='asr', prefix=prefix,ctc_policy_hypothesis=ctc_policy_hypothesis,beam_search_filtering=beam_search_filtering)
            return pred_ids, ctc_hypothesis