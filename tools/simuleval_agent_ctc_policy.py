# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy

from NMTGMinor.translate_api import add_parser_args, TranlateAPI

from simuleval.utils import entrypoint
from simuleval.agents.actions import ReadAction, WriteAction


from espnet2.bin.st_inference_streaming import Speech2TextStreaming
from espnet2.bin.st_inference_streaming import BatchBeamSearchOnline
from espnet2.utils.types import str2bool, str_or_none

BOW_PREFIX = "\u2581"
CAPITALIZING_PUNCTUATION = set(".!?")
PUNCTUATION = set(".!?,;')")

import torch
import logging

from tools.simuleval_agent23 import IWSLT23Agent

@entrypoint
class IWSLT23AgentCTCPolicy(IWSLT23Agent):

    def __init__(self, args):
        super().__init__(args)

        kwargs = vars(args)
        if kwargs['ngpu'] >= 1:
            device = "cuda"
        else:
            device = "cpu"

        org_cwd = os.getcwd()
        os.chdir(args.espnet_recipe_path)
        speech2text_kwargs = dict(
                        st_train_config=kwargs['st_train_config'],
                        st_model_file=kwargs['st_model_file'],
                        lm_train_config=kwargs['lm_train_config'],
                        lm_file=kwargs['lm_file'],
                        token_type=kwargs['token_type'],
                        bpemodel=kwargs['bpemodel'],
                        device=device,
                        maxlenratio=kwargs['maxlenratio'],
                        minlenratio=kwargs['minlenratio'],
                        dtype=kwargs['dtype'],
                        beam_size=kwargs['beam_size'],
                        ctc_weight=kwargs['ctc_weight'],
                        lm_weight=kwargs['lm_weight'],
                        penalty=kwargs['penalty'],
                        nbest=kwargs['nbest'],
                        disable_repetition_detection=kwargs['disable_repetition_detection'],
                        decoder_text_length_limit=kwargs['decoder_text_length_limit'],
                        encoded_feat_length_limit=kwargs['encoded_feat_length_limit'],
                        incremental_decode=kwargs['incremental_decode'],
                        incremental_strategy=kwargs['incremental_strategy'],
                        length_penalty=kwargs['length_penalty'],
                        ctc_finished_score=kwargs['ctc_finished_score'],
                    )
        self.speech2text = Speech2TextStreaming(**speech2text_kwargs)
        os.chdir(org_cwd)
        Bart2EspNetHypo.decoder = self.speech2text.beam_search
        Bart2EspNetHypo.smp = self.speech2text.sentence_piece
        Bart2EspNetHypo.dic = self.dic
        self.clean()

    @staticmethod
    def add_args(parser):
        IWSLT23Agent.add_args(parser)
        parser.add_argument('--espnet_recipe_path', type=str, required=True)
        parser = IWSLT23AgentCTCPolicy.add_espnet_args(parser)
        return parser

    def clean(self):
        self.hypothesis = []
        self.stable = []
        self.output = ''
        self.last_processed = 0
        self.printed = 0
        self.src = None
        self.capitalize = True
        if hasattr(self, 'speech2text') and self.speech2text is not None:
            self.speech2text.reset()

    def policy(self):
        if self._can_process():
            
            self._prepare_source()
            h = self._infer()

            if h is not None:
                logging.info(f'HYPO:   {self.model.decode(h)}')
                stable = self._policy(h)
                logging.info(f'STABLE: {self.model.decode(stable)}')
                output = self._detokenize(stable)
                self.output = ' '.join([self.output, output])
                logging.info(f'OUTPUT: {self.output}')

                if self.states.source_finished:
                    self.clean()

                if len(output) > 0:
                    return WriteAction(output, finished=self.states.source_finished)
        
        return ReadAction()

    def _infer(self):
        if self.states.source_finished or self.speech2text.extend_for_ctc_policy(self.src, False):
            src = self.src.unsqueeze(1) # T x 1
            prefix = None
            if len(self.stable) > 0:
                prefix = [self.stable]
            ctc_policy_hypothesis = None if self.states.source_finished else Bart2EspNetHypo()
            hypotheses = self.model.infer(src, prefix, ctc_policy_hypothesis=ctc_policy_hypothesis)
            hypothesis = hypotheses[0][0] # Tensor: T (no BOS + prefix)
            return list(hypothesis.cpu().numpy())
        return None
    
    @staticmethod
    def add_espnet_args(parser):
        # Note(kamo): Use '_' instead of '-' as separator.
        # '-' is confusing if written in yaml.
        parser.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )

        parser.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--dtype",
            default="float32",
            choices=["float16", "float32", "float64"],
            help="Data type",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )

        group = parser.add_argument_group("The model configuration related")
        group.add_argument(
            "--st_train_config",
            type=str,
            help="ST training configuration",
        )
        group.add_argument(
            "--st_model_file",
            type=str,
            help="ST model parameter file",
        )
        group.add_argument(
            "--lm_train_config",
            type=str,
            help="LM training configuration",
        )
        group.add_argument(
            "--src_lm_train_config",
            type=str,
            help="LM training configuration",
        )
        group.add_argument(
            "--lm_file",
            type=str,
            help="LM parameter file",
        )
        group.add_argument(
            "--src_lm_file",
            type=str,
            help="LM parameter file",
        )
        group.add_argument(
            "--word_lm_train_config",
            type=str,
            help="Word LM training configuration",
        )
        group.add_argument(
            "--src_word_lm_train_config",
            type=str,
            help="Word LM training configuration",
        )
        group.add_argument(
            "--word_lm_file",
            type=str,
            help="Word LM parameter file",
        )
        group.add_argument(
            "--src_word_lm_file",
            type=str,
            help="Word LM parameter file",
        )
        group.add_argument(
            "--ngram_file",
            type=str,
            help="N-gram parameter file",
        )
        group.add_argument(
            "--src_ngram_file",
            type=str,
            help="N-gram parameter file",
        )
        group.add_argument(
            "--model_tag",
            type=str,
            help="Pretrained model tag. If specify this option, *_train_config and "
            "*_file will be overwritten",
        )
        group.add_argument(
            "--enh_s2t_task",
            type=str2bool,
            default=False,
            help="enhancement and asr joint model",
        )

        group = parser.add_argument_group("Beam-search related")
        group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="The batch size for inference",
        )
        group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
        group.add_argument("--asr_nbest", type=int, default=1, help="Output N-best hypotheses")
        group.add_argument("--beam_size", type=int, default=20, help="Beam size")
        group.add_argument("--asr_beam_size", type=int, default=20, help="Beam size")
        group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
        group.add_argument("--asr_penalty", type=float, default=0.0, help="Insertion penalty")
        group.add_argument(
            "--maxlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain max output length. "
            "If maxlenratio=0.0 (default), it uses a end-detect "
            "function "
            "to automatically find maximum hypothesis lengths."
            "If maxlenratio<0.0, its absolute value is interpreted"
            "as a constant max output length",
        )
        group.add_argument(
            "--asr_maxlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain max output length. "
            "If maxlenratio=0.0 (default), it uses a end-detect "
            "function "
            "to automatically find maximum hypothesis lengths."
            "If maxlenratio<0.0, its absolute value is interpreted"
            "as a constant max output length",
        )
        group.add_argument(
            "--minlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain min output length",
        )
        group.add_argument(
            "--asr_minlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain min output length",
        )
        group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
        group.add_argument("--asr_lm_weight", type=float, default=1.0, help="RNNLM weight")
        group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
        group.add_argument("--asr_ngram_weight", type=float, default=0.9, help="ngram weight")
        group.add_argument("--ctc_weight", type=float, default=0.0, help="ST CTC weight")
        group.add_argument("--asr_ctc_weight", type=float, default=0.3, help="ASR CTC weight")

        group.add_argument(
            "--transducer_conf",
            default=None,
            help="The keyword arguments for transducer beam search.",
        )

        group = parser.add_argument_group("Text converter related")
        group.add_argument(
            "--token_type",
            type=str_or_none,
            default=None,
            choices=["char", "bpe", None],
            help="The token type for ST model. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--src_token_type",
            type=str_or_none,
            default=None,
            choices=["char", "bpe", None],
            help="The token type for ST model. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model path of sentencepiece. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--src_bpemodel",
            type=str_or_none,
            default=None,
            help="The model path of sentencepiece. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--ctc_greedy",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            "--sim_chunk_length",
            type=int,
            default=0,
            help="The length of one chunk, to which speech will be "
            "divided for evalution of streaming processing.",
        )
        group.add_argument("--disable_repetition_detection", type=str2bool, default=False)
        group.add_argument(
            "--encoded_feat_length_limit",
            type=int,
            default=0,
            help="Limit the lengths of the encoded feature" "to input to the decoder.",
        )
        group.add_argument(
            "--decoder_text_length_limit",
            type=int,
            default=0,
            help="Limit the lengths of the text" "to input to the decoder.",
        )

        group.add_argument(
            "--backend",
            type=str,
            default="offline",
            help="Limit the lengths of the text" "to input to the decoder.",
        )
        group.add_argument(
            "--incremental_decode",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            '--incremental-strategy',
            type=str,
            default='hold-2',
            help='Policy for selection of stable prefix for incremental decoding: hold-N (deletes last N tokens from output), local-agreement-N (Evaluates every N blocks. Takes last two generated hypotheses and outputs their common prefix).'
        )

        group.add_argument(
            '--ctc-wait',
            type=int,
            default=None,
            help='Dynamic maxlen for decoding unfinished source. Maxlen = len(greedy CTC) - ctc-wait.'
        )

        group.add_argument(
            "--detokenizer",
            type=str,
            default=None,
        )
        def none_or_float(arg):
            if arg is None or arg.lower() == 'none':
                return None
            return float(arg)
        group.add_argument(
            "--length_penalty",
            type=none_or_float,
            default=None,
            help='If None, uses length normalization.'
        )
        group.add_argument(
            "--ctc_finished_score",
            type=none_or_float,
            default=float('inf'),
            help='If None, uses length normalization.'
        )

        return parser


class Bart2EspNetHypo:
    dic = None
    smp = None
    decoder : BatchBeamSearchOnline = None

    def __init__(self,tokens=None, word=None, force_decoded=0, h=None, finished=False, score=0., orginal_tokens=0,recv_tokens=None,eos_scores=None,last_appended=1):
        self.tokens = copy.deepcopy(tokens) if tokens else []
        self.word = copy.deepcopy(word) if word else ''
        self.force_decoded = force_decoded
        if h is None:
            self.hypothesis = self.decoder.init_hyp(self.decoder.enc_for_ctc_policy)
        else:
            self.hypothesis = copy.deepcopy(h)
        self.finished = finished
        self.score = score
        self.orginal_tokens = orginal_tokens
        self.recv_tokens = copy.deepcopy(recv_tokens) if recv_tokens else ''
        self.eos_scores = copy.deepcopy(eos_scores) if eos_scores else []
        self.last_appended = last_appended

    def copy(self):
        return Bart2EspNetHypo(
            self.tokens,
            self.word,
            self.force_decoded,
            self.hypothesis,
            self.finished,
            self.score,
            self.orginal_tokens,
            self.recv_tokens,
            self.eos_scores,
            self.last_appended,
        )

    def append_tokens(self, tokens):
        for token in tokens[self.last_appended:].cpu().numpy():
            token = self.dic[token]
            if token.startswith('▁') and len(self.word) > 1:
                self.tokens += self.smp.EncodeAsIds(self.word)
                self.recv_tokens += ''.join(self.smp.EncodeAsPieces(self.word))
                self.word = ''
                self.hypothesis, finished, eos_scores = self.decoder._ctc_force_decode_tokens(self.hypothesis, self.tokens, self.force_decoded, len(self.tokens))
                self.finished |= finished
                self.force_decoded = len(self.tokens)
                self.orginal_tokens = 0
                self.eos_scores += eos_scores
            self.orginal_tokens += 1
            self.word += token
        self.last_appended = tokens.shape[0]
    
    def is_finished(self):
        return self.finished