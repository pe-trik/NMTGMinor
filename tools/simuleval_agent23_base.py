# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from NMTGMinor.translate_api import add_parser_args, TranlateAPI

from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction

from mosestokenizer import MosesDetokenizer

BOW_PREFIX = "\u2581"
CAPITALIZING_PUNCTUATION = set(".!?")
PUNCTUATION = set(".!?,;')")

import torch
import logging

class IWSLT23AgentBase(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        kwargs = vars(args)
    
        logging.basicConfig(
            level=kwargs['log_level'],
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )


        self.args = args
        self.model = TranlateAPI(args)
        self.dic = self.model.dict()
        self.chunk_len = self.args.chunk_len
        self.hold = args.hold
        self.la = args.local_agreement
        self.beam_search_filtering = args.beam_search_filtering
        self.character_level = args.eval_latency_unit == 'char'
        self.chinese = args.sacrebleu_tokenizer == 'zh'
        self.japanese = args.sacrebleu_tokenizer == 'ja-mecab'

        self.detokenizer = MosesDetokenizer(args.lang)
        self.clean()

    @staticmethod
    def add_args(parser):
        add_parser_args(parser)
        parser.add_argument('--chunk_len', type=int, default=800)
        parser.add_argument('--hold', type=int, default=-1)
        parser.add_argument('--local_agreement', type=int, default=-1)
        parser.add_argument('--lang', type=str, default='de')
        parser.add_argument('--beam_search_filtering', action='store_true')
        return parser

    def clean(self):
        self.hypothesis = []
        self.stable = []
        self.output = ''
        self.last_processed = 0
        self.printed = 0
        self.src = None
        self.capitalize = True

    def policy(self):
        if self._can_process():
            self._prepare_source()
            h = self._infer()
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

    def _detokenize(self, stable):
        self.stable = stable
        new_printed = self.printed
        if self.states.source_finished or self.character_level:
            new_printed = len(stable)
        else:
            for i in range(self.printed, len(stable)):
                token = self.dic[stable[i]]
                if token[0] == BOW_PREFIX:
                    if i > 0 and '(' in self.dic[stable[i - 1]]:
                        continue
                    if any(c.isalnum() for c in token):
                        new_printed = i
                    if any(c in PUNCTUATION for c in token) and not any(c.isdigit() for c in token):
                        new_printed = i + 1
        if new_printed > self.printed:
            output = [self.dic[t] for t in stable[self.printed:new_printed]]
            output = ''.join(output).split(BOW_PREFIX)
            output = self.detokenizer(output)
            if self.chinese:
                output = output.replace(',', '，')
                output = output.replace('?','？')
            elif self.japanese:
                output = output.replace('?', '？')
            output = list(output)
            for idx, c in enumerate(output):
                self.capitalize |= c in CAPITALIZING_PUNCTUATION
                if self.capitalize and c.lower() != c.upper():
                    self.capitalize = False
                    output[idx] = c.upper()
            output = ''.join(output)
            self.printed = new_printed
            return output
        return ''

    def _policy(self, new_hypothesis):
        stable = len(self.stable)
        while len(new_hypothesis) > 0 and new_hypothesis[-1] == self.model.EOS:
            new_hypothesis = new_hypothesis[:-1]
        if not self.states.source_finished:
            if self.hold > -1:
                stable = max(stable, len(new_hypothesis) - self.hold)
            if self.la > -1:
                for idx, (p, n) in enumerate(zip(self.hypothesis, new_hypothesis)):
                    stable = idx - 1
                    if p != n:
                        break
        else:
            stable = len(new_hypothesis)
        stable = max(stable, len(self.stable))
        self.hypothesis = new_hypothesis
        return new_hypothesis[:stable]

    def _infer(self):
        src = self.src.unsqueeze(1) # T x 1
        prefix = None
        if len(self.stable) > 0:
            prefix = [self.stable]
        hypotheses, _ = self.model.infer(src, prefix, beam_search_filtering=self.beam_search_filtering and not self.states.source_finished)
        hypothesis = hypotheses[0][0] # Tensor: T (no BOS + prefix)
        return list(hypothesis.cpu().numpy())

    def _can_process(self):
        return self.states.source_finished or len(self.states.source) / 16 >= self.last_processed / 16 + self.chunk_len
        
    def _prepare_source(self):
        if self.src is not None:
            self.src = torch.concat(
                (self.src, torch.tensor(self.states.source[self.src.size(0):])),
                0,
            )
        else:
            self.src = torch.tensor(self.states.source)
        self.last_processed = len(self.states.source)
        return self.src
    


