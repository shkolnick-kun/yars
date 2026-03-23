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
from scipy.optimize import minimize

#==============================================================================
import os
import sys

SRC_PATH = os.path.dirname(os.path.dirname(__file__))

#------------------------------------------------------------------------------
# Add filter design things to path
FD_PATH = os.path.join(SRC_PATH, 'filter_design')
sys.path.append(FD_PATH)
from make_filter import make_filter

#------------------------------------------------------------------------------
# Build and import extension
import pyximport
EXT_PATH = os.path.join(os.path.join(SRC_PATH, 'src'), 'yarspy')
sys.path.insert(0, EXT_PATH)
pyximport.install(pyimport=True, reload_support=True)
import yarspy

#==============================================================================

taps = 79
attenuation = 100.0
oversample = 2000
max_degree = 16
tolerance = 5e-6

config, stop_atten, stop_start, minus3db, y_test, x_test = make_filter(
    taps, attenuation, oversample, max_degree, tolerance
)

y = yarspy.weight(x_test, config)

def error(x):
    return np.max(np.abs(y_test.astype(np.float64) - y.astype(np.float64) * x[0]))

res = minimize(error, np.array([1.0]))

config['polly'] *= res.x[0]
#config['polly'][-1] += res.x[1]

y = yarspy.weight(x_test, config)
e = y_test - y

print("\nValidation errors:")
print(np.max(e), np.min(e))

print("\n#   Corrected config:")
print(f"#   length           : {config['ntaps']}")
print(f"#   Fudge factor     : {config['fudge']:.10e}")
print(f"#   Window factor    : {config['window']:.10e}")
print(f"#   Polly            : {len(config['polly'])}")
for i in range(len(config['polly'])):
    print(f"#   [{i}] = {config['polly'][i]:.10e}")

plt.plot(x_test, y, x_test, y_test)
plt.show()

plt.plot(x_test, e, x_test, y * np.max(np.abs(e)))
plt.show()

print("\nFinal errors:")
print(np.max(e), np.min(e))
