#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for asking the local Soprano QA system."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from soprano_qa.answer import main


if __name__ == "__main__":
    main()
