# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os

from .model import Model
from nmtg.models.encoder_decoder import EncoderDecoderModel

MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a new Model."""

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate Model ({})'.format(name))
        if not issubclass(cls, Model):
            raise ValueError('Model ({}: {}) must extend Model'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('nmtg.models.' + module)


def get_model_type(name):
    return MODEL_REGISTRY[name]


def get_model_names():
    return list(MODEL_REGISTRY.keys())
