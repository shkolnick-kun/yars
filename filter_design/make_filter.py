#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter design module for a windowed‑sinc resampler.

This module provides functions to design a low‑pass FIR filter using a
windowed sinc kernel. The window is a Kaiser window approximated by a
polynomial (even function). The design parameters are adjusted so that
the stop‑band edge matches the target frequency.
"""
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
import mpmath as mp
from scipy.fft import fft
import matplotlib.pyplot as plt
from approx import chebyshev_approximation_even

SPEC_LEN = 1000000


def generate_filter(taps, fudge_factor, oversample, polly):
    """
    Generate the impulse response of a low‑pass windowed sinc filter.

    The filter length is determined by the number of taps and the oversampling
    factor. The window is evaluated from the polynomial coefficients.

    Args:
        taps (int): Number of filter taps (must be odd for a symmetric filter).
        fudge_factor (float): Scaling factor for the sinc function.
        oversample (int): Oversampling factor used for filter design.
        polly (numpy.ndarray): Polynomial coefficients in descending order
            (e.g., [a_n, a_{n-1}, ..., a_0]) evaluated at (x²) to obtain
            the window value.

    Returns:
        tuple:
            f (numpy.ndarray): Impulse response of the filter (normalised).
            m (numpy.ndarray): Index array used for the sinc function.
            wf (float): Window factor = 2 * oversample / N, where N is the
                number of points in the generated filter.
    """
    # Number of points in the generated filter (must be odd)
    if taps % 2 == 0:
        N = int(taps * oversample + 1)
    else:
        N = int((taps - 1) * oversample + 1)

    # Indices for the sinc function (centred)
    m = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1.0)

    # Sinc kernel (ideal low‑pass)
    f = np.sinc(m / fudge_factor / oversample)

    # Window factor and normalised argument for the polynomial
    wf = 2.0 * oversample / N
    arg = (m / oversample) * wf          # in [-1, 1]
    window_val = np.polyval(polly, arg * arg)   # polynomial in (arg²)

    # Apply window and normalise to unit sum
    f = f * window_val
    f = f / np.sum(f)

    return f, m / oversample, wf


def measure_filter(f, atten):
    """
    Measure the frequency response of a filter and extract key parameters.

    The stop‑band attenuation, the start of the stop‑band, and the -3 dB
    point are estimated from the FFT of the impulse response.

    Args:
        f (numpy.ndarray): Impulse response (real coefficients).
        atten (float): Desired stop‑band attenuation (dB). Used to locate
            the stop‑band region.

    Returns:
        tuple:
            stop_atten (float): Achieved stop‑band attenuation (dB).
            stop_band_start (float): Normalised frequency (0..0.5) where the
                stop‑band begins (first local minimum below -0.8*atten).
            minus_3db (float): Normalised frequency where the magnitude
                response drops by 3 dB.
    """
    # Zero‑pad and compute the FFT
    padded = np.zeros(SPEC_LEN, dtype=complex)
    padded[:len(f)] = f
    spec = fft(padded)
    spec = 20 * np.log10(np.abs(spec[:SPEC_LEN // 2]) + 1e-15)

    # Find the first local minimum below -0.8*atten (start of stop‑band)
    first_null = 0
    for k in range(1, len(spec)):
        if (spec[k] < -0.8 * atten and
            spec[k-1] > spec[k] and
            spec[k] < spec[k+1]):
            first_null = k
            break

    # Maximum attenuation in the stop‑band
    stop_atten = np.max(spec[first_null:])

    # Locate the crossing of the stop‑band attenuation level in the
    # transition band.
    atten_start = 0
    for k in range(first_null):
        if spec[k] > stop_atten and spec[k+1] < stop_atten:
            atten_start = k
            break

    # Interpolate the stop‑band start frequency
    stop_band_start = (atten_start +
                       (stop_atten - spec[atten_start]) /
                       (spec[atten_start+1] - spec[atten_start]))
    stop_band_start /= SPEC_LEN

    # Find the -3 dB point
    minus_3db = 0
    for k in range(first_null):
        if spec[k] > -3.0 and spec[k+1] < -3.0:
            minus_3db = k
            break

    # Interpolate the -3 dB frequency
    minus_3db_interp = (minus_3db +
                        (-3.0 - spec[minus_3db]) /
                        (spec[minus_3db+1] - spec[minus_3db]))
    minus_3db_interp /= SPEC_LEN

    return stop_atten, stop_band_start, minus_3db_interp


def plot_filter(f, atten, stop_band_start, minus3db, target, filename=None):
    """
    Plot the frequency response of the filter.

    The frequency axis is normalised so that 1 corresponds to the pass‑band
    edge (target frequency). The plot highlights the pass‑band, stop‑band,
    and the -3 dB point.

    Args:
        f (numpy.ndarray): Impulse response of the filter.
        atten (float): Desired stop‑band attenuation (dB).
        stop_band_start (float): Normalised stop‑band start frequency
            (obtained from measure_filter).
        minus3db (float): Normalised -3 dB frequency.
        target (float): Target normalised pass‑band edge (0.5/oversample).
        filename (str, optional): If provided, the plot is saved to this file.
            Otherwise, the figure object is returned.

    Returns:
        matplotlib.figure.Figure: The figure object (if filename is None).
    """
    # Compute spectrum
    padded = np.zeros(SPEC_LEN, dtype=complex)
    padded[:len(f)] = f
    spec = fft(padded)
    spec_db = 20 * np.log10(np.abs(spec[:SPEC_LEN // 2]) + 1e-15)
    freq = np.linspace(0, 0.5, SPEC_LEN // 2)

    # Normalise frequency to target (target = 0.5/oversample)
    target *= 2.0
    freq_norm = freq / target

    print("\n# Frequency response:")
    current = 0.025
    for i, f_val in enumerate(freq_norm):
        if f_val >= current:
            print(f"#   freq = {2.0 * f_val:.3f}, level = {spec_db[i]:.2f}")
            current += 0.025
            if current > 0.6:
                break

    stop_band_start_norm = stop_band_start / target
    minus3db_norm = minus3db / target

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_norm, spec_db, label='Frequency response')
    ax.axvline(x=stop_band_start_norm, color='r', linestyle='--',
               label='Stop‑band start')
    ax.axvline(x=minus3db_norm, color='g', linestyle='--',
               label='-3 dB point')
    ax.axhline(y=-3, color='g', linestyle=':', alpha=0.5)
    ax.axhline(y=-atten, color='r', linestyle=':', alpha=0.5,
               label=f'Target attenuation {atten} dB')
    ax.axvspan(stop_band_start_norm, 1, alpha=0.2, color='grey',
               label='Stop‑band')
    ax.set_xlabel('Normalised frequency (relative to 1/oversample)')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_title('Filter frequency response')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, 0.6)

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        return None
    else:
        return fig


def make_filter(taps, attenuation, oversample=1000, max_polly_degree=16,
                tolerance=1e-6):
    """
    Design a low‑pass filter for a windowed‑sinc resampler.

    The filter uses a Kaiser window approximated by a Chebyshev polynomial
    (even function). The fudge factor is adjusted so that the stop‑band edge
    equals 0.5/oversample. The resulting configuration is suitable for the
    Resampler class.

    Args:
        taps (int): Number of filter taps (ntaps). Should be odd.
        attenuation (float): Desired stop‑band attenuation (dB).
        oversample (int): Oversampling factor used in design (default 1000).
        max_polly_degree (int): Maximum polynomial degree in the variable y
            (where the polynomial is evaluated at y²). Default 16.
        tolerance (float): Maximum absolute error allowed in the polynomial
            approximation of the Kaiser window. Default 1e-6.

    Returns:
        tuple:
            config (dict): Resampler configuration containing:
                - 'fudge' (float): fudge factor.
                - 'window' (float): window factor.
                - 'ntaps' (int): number of filter taps.
                - 'polly' (numpy.ndarray): polynomial coefficients in
                  descending order (float32).
            stop_atten (float): Achieved stop‑band attenuation (dB).
            stop_start (float): Normalised stop‑band start frequency.
            minus3db (float): Normalised -3 dB frequency.
            f_scaled (numpy.ndarray): Scaled impulse response (f * oversample).
            m (numpy.ndarray): Index array used for the sinc function.
    """
    # ----------------------------------------------------------------------
    # Compute polynomial approximation of the Kaiser window
    # ----------------------------------------------------------------------
    beta = (attenuation + 0.5) / 10.0
    I0_beta = mp.besseli(0, beta)

    # Kaiser window on [-1,1]: w(x) = I0(beta*sqrt(1-x²)) / I0(beta)
    def window_func(x):
        arg = beta * mp.sqrt(1 - x * x)
        return mp.besseli(0, arg) / I0_beta

    # Obtain Chebyshev approximation coefficients (lowest power first)
    # The function is even, so we use chebyshev_approximation_even.
    _, coeffs, _, _ = chebyshev_approximation_even(
        window_func, max_degree=max_polly_degree, tolerance=tolerance
    )
    # Convert to float32 and reverse order (highest power first)
    polly = coeffs[::-2].astype(np.float32)

    # ----------------------------------------------------------------------
    # Adjust fudge_factor so that the stop‑band edge matches the target
    # ----------------------------------------------------------------------
    # Initial guesses
    fudge1 = 1.0
    f1, m1, wf = generate_filter(taps, fudge1, oversample, polly)
    stop_atten1, stop_start1, minus3db1 = measure_filter(f1, attenuation)

    fudge2 = 1.25
    f2, m2, wf = generate_filter(taps, fudge2, oversample, polly)
    stop_atten2, stop_start2, minus3db2 = measure_filter(f2, attenuation)

    print(f"    fudge_factor: {fudge1:.10f}   stop_band_start: {stop_start1:.10f}   1")
    print(f"    fudge_factor: {fudge2:.10f}   stop_band_start: {stop_start2:.10f}   2")

    f = f1
    stop_atten = stop_atten1
    stop_start = stop_start1
    minus3db = minus3db1
    fudge = fudge1
    target = 0.5 / oversample

    # Binary search to bring stop_band_start to target
    while (stop_start1 - stop_start2) > 1e-10:
        fudge = (fudge1 + fudge2) / 2
        f, m, wf = generate_filter(taps, fudge, oversample, polly)
        stop_atten, stop_start, minus3db = measure_filter(f, attenuation)

        if stop_start1 < stop_start2:
            print("stop_band_start1 < stop_band_start2 during iteration")
            break

        if (np.abs(fudge2 - fudge1) / 2) < (fudge * 2 * np.finfo(fudge).eps):
            print("No changes in fudge")
            break

        if stop_start > 1.0:
            raise ValueError(f"    Fail: {fudge1:.10f}   {fudge:.10f}   {fudge2:.10f}")

        if stop_start <= target:
            # Update right bound (value less than target)
            f2 = f
            stop_start2 = stop_start
            fudge2 = fudge
            choice = 2
        else:
            # Update left bound
            f1 = f
            stop_start1 = stop_start
            fudge1 = fudge
            choice = 1

        print(f"    fudge_factor: {fudge:.10f}   stop_band_start: {stop_start:.10f}   {choice}  delta: {stop_start1-stop_start2:.3e}")

    # Output statistics
    N = len(f)
    print(f"\n# f = make_filter({taps}, {attenuation}, oversample={oversample}) ;")
    print(f"#   Pass band width  : {stop_start * oversample * 2:.10f} (should be {1.0:.10f})")
    print(f"#   Stop band atten. : {abs(stop_atten):.2f} dB")
    print(f"#   -3dB band width  : {oversample * minus3db / 0.5:.3f}")
    print(f"#   length           : {taps}")
    print(f"#   Fudge factor     : {fudge:.10e}")
    print(f"#   Window factor    : {wf:.10e}")
    print(f"#   Polly            : {len(polly)}")
    for i in range(len(polly)):
        print(f"#   [{i}] = {polly[i]:.10e}")


    # Scaled version of the filter (useful for plotting)
    f_scaled = f * oversample

    # Build configuration dictionary for the Resampler class
    config = {
        'fudge': float(fudge),
        'window': float(wf),
        'ntaps': int(taps),
        'polly': polly
    }

    return config, stop_atten, stop_start, minus3db, f_scaled, m


if __name__ == "__main__":
    # Example usage
    taps = 79
    attenuation = 100.0
    oversample = 2000
    max_degree = 16
    tolerance = 5e-6

    config, stop_atten, stop_start, minus3db, f, m = make_filter(
        taps, attenuation, oversample, max_degree, tolerance
    )

    # Plot the frequency response
    target = 0.5 / oversample
    plot_filter(f / oversample, attenuation, stop_start, minus3db, target,
                filename=None)
    plt.show()
