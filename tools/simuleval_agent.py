# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from NMTGMinor.translate_api import add_parser_args, TranlateAPI

from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction

BOW_PREFIX = "\u2581"

import torch
import logging

@entrypoint
class DummyAgent(SpeechToTextAgent):
    """
    DummyAgent operates in an offline mode.
    Waits until all source is read to run inference.
    """

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

        self.clean()

    @staticmethod
    def add_args(parser):
        add_parser_args(parser)
        parser.add_argument('--chunk_len', type=int, default=800)
        return parser

    def clean(self):
        self.stable = []
        self.last_processed = 0
        self.printed = 0

    def policy(self):
        if self._can_process():
            src = self._prepare_source()
            prefix = [torch.LongTensor(self.stable)]
            logging.info("Displayed" + self.model.decode(self.stable))
            hypo, _ = self.model.infer(src, prefix, 'wav')
            hypo = list(hypo[0][0].cpu().numpy())

            logging.info("Hypo" + self.model.decode(hypo))

            if not self.states.source_finished:
                hypo = hypo[:-2]

            self.stable += hypo
            printable = self._num_printable()

            if printable > self.printed or self.states.source_finished:
                o = ''.join([self.dic[t] for t in self.stable[self.printed:printable]]).replace(BOW_PREFIX, ' ')
                self.printed = printable
                if self.states.source_finished:
                    self.clean()
                return WriteAction(o, finished=self.states.source_finished)
        
        return ReadAction()

    def _can_process(self):
        return self.states.source_finished or len(self.states.source) / 16 >= self.last_processed / 16 + self.chunk_len
        
    def _prepare_source(self):
        src = torch.tensor(self.states.source)
        src /= (1 << 31)
        src /= src.abs().max()
        src = src.unsqueeze(1)
        self.last_processed = len(self.states.source)
        return src

    def _num_printable(self):
        if self.states.source_finished:
            return len(self.states.source)
        printable_idx = -1
        for idx, token in enumerate(self.stable):
            token = self.dic[token]
            if token.endswith(BOW_PREFIX) and len(token) > 1:
                printable_idx = idx
            if token.startswith(BOW_PREFIX):
                printable_idx = idx - 1
        return printable_idx + 1

