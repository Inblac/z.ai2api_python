#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib


def __getattr__(name):
    if name == "provider":
        return importlib.import_module("app.providers.zai.provider")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
