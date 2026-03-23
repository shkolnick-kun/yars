#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Copyright 2026 anonimous <shkolnick-kun@gmail.com> and contributors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

    See the License for the specific language governing permissions
    and limitations under the License.
"""
import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
# import os
# import sys

# SRC_PATH = os.path.dirname(os.path.dirname(__file__))

#------------------------------------------------------------------------------
# Add filter design things to path
# FD_PATH = os.path.join(SRC_PATH, 'filter_design')
# sys.path.append(FD_PATH)
# from make_filter2 import make_filter

#------------------------------------------------------------------------------
# Build and import extension
# import pyximport
# EXT_PATH = os.path.join(os.path.join(SRC_PATH, 'src'), 'yarspy')
# sys.path.insert(0, EXT_PATH)
# pyximport.install(pyimport=True, reload_support=True)
import yarspy

#==============================================================================
x_test = np.linspace(0, 40, 4000, dtype=np.float32)
y_test = np.sinc(x_test)

y = yarspy.sinc(x_test)

e = y - y_test

print(np.max(e), np.min(e))

plt.plot(x_test, y_test, 'green', x_test, y, 'red',)
plt.show()

plt.plot(x_test, e)
plt.show()
