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
import os
import sys

SRC_PATH = os.path.dirname(os.path.dirname(__file__))

#------------------------------------------------------------------------------
# Add filter design things to path
FD_PATH = os.path.join(SRC_PATH, 'doc')
sys.path.append(FD_PATH)
from prototype import DEFAULT_CFG

#------------------------------------------------------------------------------
# Build and import extension
import pyximport
EXT_PATH = os.path.join(os.path.join(SRC_PATH, 'src'), 'yarspy')
sys.path.insert(0, EXT_PATH)
pyximport.install(pyimport=True, reload_support=True)
import yarspy

#==============================================================================

N = 1000

f  = 8.
px = np.array([0.001 * i for i in range(N)]) * f
x = np.sin(2 * np.pi * px)

cur = 0
def _input_cb(*args):
    global cur
    if cur < len(x):
        ret = x[cur]
    else:
        ret = 0
    cur += 1
    return ret

ratio = 1 / np.pi / 2

resample_x = yarspy.Resampler(_input_cb, ratio, DEFAULT_CFG)

yl = []
while cur < len(x):
    yl.append(resample_x())

y = np.array(yl)
py = np.array([0.001 * i for i in range(len(y))]) * f / ratio

plt.plot(px, x, '.-', label='Input')
plt.plot(py, y, '*-', label='Resampled')
plt.legend()
plt.show()
