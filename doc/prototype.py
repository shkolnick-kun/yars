#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2026 anonimous <shkolnick-kun@gmail.com> and contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Resampler module: a sample rate converter using a windowed sinc interpolator.
The filter coefficients are defined by a polynomial that approximates a Kaiser
window with 100 dB sidelobe attenuation. The implementation follows the same
API as the C version with Cython bindings.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Default configuration (matches the C implementation)
# -----------------------------------------------------------------------------
DEFAULT_CFG = {
    'fudge': 1.08947476744651794434,
    'window': 0.02564069691414212759,
    'ntaps': 79,
    # Polynomial coefficients in descending order (suitable for Horner's scheme).
    'polly': np.array([
        0.13379795737644664677,  # degree 16
        -0.93236506483873515805, # degree 14
        3.199451818376378931,    # degree 12
        -7.1954835324181223299,  # degree 10
        11.54754684064062964,    # degree  8
        -13.224591050031412109,  # degree  6
        10.23997209407132658,    # degree  4
        -4.7679898064693988502,  # degree  2
        1.0000011                # degree  0 (tuned to minimise maximum error)
    ], dtype=np.float32)
}


# -----------------------------------------------------------------------------
def _rsm_sinc(x):
    """Approximate sinc(x) using a polynomial.

    The function implements the approximation used in the original resampler.

    Args:
        x (float or numpy.ndarray): Input value(s).

    Returns:
        float or numpy.ndarray: Approximation of sinc(x) = sin(πx)/(πx).

    Note:
        For scalar input the result is a scalar; for array input an array is
        returned.
    """
    # Coefficients in descending order (Horner scheme).
    C = np.array([
        -0.0019068844782283160717,  # degree 10
        0.02561632606712057128,     # degree  8
        -0.19043806034253935033,    # degree  6
        0.81165180125819180734,     # degree  4
        -1.6449228124140149454,     # degree  2
        0.99999959992067255499      # degree  0
    ], dtype=np.float32)

    ix = np.ceil(0.5 * (x - 1.0))
    fx = x - 2 * ix
    out = np.polyval(C, fx * fx)

    if isinstance(x, np.ndarray):
        ids = ix != 0
        out[ids] *= fx[ids] / x[ids]
        return out
    else:
        if ix != 0:
            return out * fx / x
        else:
            return out


# -----------------------------------------------------------------------------
class Resampler:
    """Sample rate converter using a windowed sinc interpolation filter.

    The filter is defined by a configuration dictionary that specifies the
    polynomial coefficients for the window, the fudge factor for the sinc,
    the window scale factor and the number of taps (filter length).

    Attributes:
        fudge (float): Sinc scaling factor.
        window (float): Window scaling factor.
        ntaps (int): Number of filter taps (ring buffer length).
        polly (numpy.ndarray): Polynomial coefficients for the Kaiser window
            approximation (descending order).
        input_cb (callable): Callback that provides the next input sample.
        ratio (float): Resampling ratio (output sample rate / input sample rate).
        ring (numpy.ndarray): Circular buffer for past input samples.
        index (int): Current write position in the ring buffer.
        phase (float): Internal phase accumulator (fractional output sample index).
    """

    def __init__(self, input_cb, ratio, cfg=DEFAULT_CFG):
        """Initialise the resampler.

        Args:
            input_cb (callable): A callable that returns the next input sample
                (float) each time it is called. It will be called repeatedly
                as needed.
            ratio (float): Desired resampling ratio = f_out / f_in.
            cfg (dict, optional): Configuration dictionary. If not provided,
                the default configuration (DEFAULT_CFG) is used. The dictionary
                must contain the keys 'fudge', 'window', 'ntaps', and 'polly'.

        Raises:
            ValueError: If input_cb is not callable.
        """
        if not callable(input_cb):
            raise ValueError("input_cb must be callable!")

        # Load configuration.
        self.fudge = float(cfg['fudge'])
        self.window = float(cfg['window'])
        self.ntaps = int(cfg['ntaps'])
        self.polly = np.asarray(cfg['polly'], dtype=np.float32)

        # Ensure the ratio is at least the minimum required value.
        min_ratio = 2.0 / self.ntaps
        if ratio < min_ratio:
            print(f"Warning: ratio must be >= {min_ratio:.4e}. Using {min_ratio:.4e}.")
            ratio = min_ratio

        self.input_cb = input_cb
        self.ratio = ratio
        self.ring = np.zeros(self.ntaps, dtype=np.float32)
        self.index = 0
        self.phase = 0.0

    def _weight(self, x):
        """Compute the filter weight for a given fractional phase offset.

        The weight is the product of the sinc approximation and the window
        polynomial, scaled by the fudge factor.

        Args:
            x (float): Phase offset (relative to the centre of the filter).

        Returns:
            float: Filter coefficient for the given phase.
        """
        sinc_val = _rsm_sinc(x / self.fudge)
        y = x * self.window
        # Evaluate the polynomial at (y^2). The coefficients are in descending
        # order, so we can directly pass them to np.polyval.
        window_val = np.polyval(self.polly, y * y)
        return sinc_val * window_val / self.fudge

    def __call__(self):
        """Generate the next output sample.

        The method fetches new input samples as needed, computes the convolution
        of the filter with the current ring buffer, updates the internal phase,
        and returns the interpolated sample.

        Returns:
            float: The next output sample.
        """
        # Fill the ring buffer with enough new samples to satisfy the phase.
        while self.phase > 0:
            self.index += 1
            if self.index >= self.ntaps:
                self.index = 0
            self.ring[self.index] = self.input_cb()
            self.phase -= 1

        # Compute the interpolated output.
        # filter_phase is the phase of the first filter tap relative to the
        # centre of the filter.
        filter_phase = self.phase - (self.ntaps - 1) / 2.0 + 1.0
        j = self.index
        out = 0.0
        for _ in range(self.ntaps):
            out += self._weight(filter_phase) * self.ring[j]
            filter_phase += 1.0
            j -= 1
            if j < 0:
                j = self.ntaps - 1

        # Advance the output phase.
        self.phase += 1.0 / self.ratio
        return out


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple test with a sine wave.
    import matplotlib.pyplot as plt

    N = 1000
    f = 8.0
    px = np.array([0.001 * i for i in range(N)]) * f
    x = np.sin(2 * np.pi * px)

    cur = 0

    def _input_cb():
        global cur
        if cur < len(x):
            ret = x[cur]
        else:
            ret = 0
        cur += 1
        return ret

    # Use a ratio slightly above the minimum to avoid clipping.
    ratio = 15 / DEFAULT_CFG['ntaps']

    resample = Resampler(_input_cb, ratio)

    y = []
    while cur < len(x):
        y.append(resample())

    y = np.array(y)
    py = np.array([0.001 * i for i in range(len(y))]) * f / ratio

    plt.plot(px, x, '.-', label='Input')
    plt.plot(py, y, '*-', label='Resampled')
    plt.legend()
    plt.show()
