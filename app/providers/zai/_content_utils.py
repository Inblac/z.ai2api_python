#!/usr/bin/env python
# -*- coding: utf-8 -*-


def clean_thinking_content(delta_content: str) -> str:
    if delta_content.startswith("<details") and "</summary>\n>" in delta_content:
        return delta_content.split("</summary>\n>")[-1].strip()
    return delta_content
