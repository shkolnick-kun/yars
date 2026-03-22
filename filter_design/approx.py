#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for Chebyshev polynomial approximation of functions on [-1, 1].

Provides three functions for approximation (general, even, odd) with automatic
degree selection based on validation error.
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

import mpmath as mp
import numpy as np


# ----------------------------------------------------------------------
# Helper functions for polynomial manipulation
# ----------------------------------------------------------------------

def _cheb2poly_mp(coeffs_cheb):
    """Convert Chebyshev coefficients to standard polynomial coefficients.

    Args:
        coeffs_cheb (list of mpf): Chebyshev coefficients in the form
            [a0/2, a1, a2, ..., an].

    Returns:
        list of mpf: Standard polynomial coefficients c0 + c1*x + ... + cn*x^n.
    """
    n = len(coeffs_cheb) - 1
    if n < 0:
        return []

    # Coefficients of Chebyshev polynomials T_k(x) (each as list of coefficients from lowest power)
    T_coeffs = []
    # T0
    T_coeffs.append([mp.mpf(1)])
    if n >= 1:
        # T1
        T_coeffs.append([mp.mpf(0), mp.mpf(1)])

    # Compute T_k for k = 2..n using recurrence
    for k in range(2, n + 1):
        prev2 = T_coeffs[k - 2]   # T_{k-2} length k-1
        prev1 = T_coeffs[k - 1]   # T_{k-1} length k
        # shift prev1 to the right by 1 (multiply by x)
        shift_prev1 = [mp.mpf(0)] + prev1   # length k+1
        two_shift = [2 * c for c in shift_prev1]  # 2 * x * T_{k-1}
        # pad prev2 with zeros to length k+1
        prev2_padded = prev2 + [mp.mpf(0)] * (k + 1 - len(prev2))
        # T_k = 2*x*T_{k-1} - T_{k-2}
        Tk = [two_shift[i] - prev2_padded[i] for i in range(k + 1)]
        T_coeffs.append(Tk)

    # Assemble final polynomial coefficients
    poly_coeffs = [mp.mpf(0) for _ in range(n + 1)]
    for k in range(n + 1):
        a_k = coeffs_cheb[k]
        Tk = T_coeffs[k]  # length k+1
        for i, coeff_t in enumerate(Tk):
            poly_coeffs[i] += a_k * coeff_t

    return poly_coeffs


def _eval_poly(coeffs_float, x):
    """Evaluate polynomial at given point using NumPy.

    Args:
        coeffs_float (np.ndarray): Coefficients from lowest to highest power.
        x (float): Point where to evaluate.

    Returns:
        float: Value of the polynomial at x.
    """
    return np.polyval(coeffs_float[::-1], x)


def _generate_validation_points(n_val):
    """Generate uniformly spaced points on [-1, 1].

    Args:
        n_val (int): Number of points.

    Returns:
        np.ndarray: Array of points.
    """
    return np.linspace(-1.0, 1.0, n_val)


def _compute_errors_for_degree(deg, full_coeffs_cheb, x_test, f):
    """Compute approximation errors for a given degree.

    Args:
        deg (int): Degree of the Chebyshev polynomial.
        full_coeffs_cheb (list of mpf): Precomputed Chebyshev coefficients up to max_degree.
        x_test (np.ndarray): Validation points.
        f (callable): Original function (expects mpf argument, returns mpf).

    Returns:
        tuple:
            - max_error (float): Maximum absolute error on validation set.
            - coeffs_float (np.ndarray): Standard polynomial coefficients (float64).
            - errors (np.ndarray): Absolute errors at validation points.
    """
    coeffs_cheb = full_coeffs_cheb[:deg + 1]
    poly_coeffs_mp = _cheb2poly_mp([coeffs_cheb[0] / 2] + coeffs_cheb[1:])
    poly_coeffs_float = np.array([float(c) for c in poly_coeffs_mp], dtype=np.float64)

    errors = np.zeros(len(x_test))
    for i, x in enumerate(x_test):
        approx_val = _eval_poly(poly_coeffs_float, x)
        exact_val = f(mp.mpf(x))
        if isinstance(exact_val, mp.mpf):
            exact_val = float(exact_val)
        errors[i] = exact_val - approx_val

    return np.max(np.abs(errors)), poly_coeffs_float, errors


def _find_best_degree(max_degree, full_coeffs_cheb, x_test, f, tolerance):
    """Find the smallest degree satisfying the tolerance.

    Args:
        max_degree (int): Maximum degree to consider.
        full_coeffs_cheb (list of mpf): Precomputed Chebyshev coefficients up to max_degree.
        x_test (np.ndarray): Validation points.
        f (callable): Original function (expects mpf argument, returns mpf).
        tolerance (float): Allowed maximum absolute error.

    Returns:
        tuple:
            - best_deg (int): Selected degree.
            - best_coeffs (np.ndarray): Standard polynomial coefficients (float64).
            - best_errors (np.ndarray): Errors at validation points for selected degree.
            - best_max_error (float): Maximum error for selected degree.
    """
    best_deg = None
    best_coeffs = None
    best_errors = None
    best_max_error = None

    for deg in range(max_degree, -1, -1):
        max_err, coeffs_float, errors = _compute_errors_for_degree(
            deg, full_coeffs_cheb, x_test, f
        )
        if max_err <= tolerance:
            best_deg = deg
            best_coeffs = coeffs_float
            best_errors = errors
            best_max_error = max_err
        else:
            # Once we exceed tolerance, lower degrees will have larger error,
            # so we stop.
            break

    if best_deg is None:
        # No degree satisfied the tolerance; fall back to max_degree.
        best_deg = max_degree
        best_max_error, best_coeffs, best_errors = _compute_errors_for_degree(
            max_degree, full_coeffs_cheb, x_test, f
        )
        print(
            f"Warning: tolerance {tolerance} not achieved for any degree up to {max_degree}. "
            f"Using degree {max_degree} with max error {best_max_error:.2e}"
        )

    return best_deg, best_coeffs, best_errors, best_max_error


# ----------------------------------------------------------------------
# Main approximation functions
# ----------------------------------------------------------------------

def chebyshev_approximation(f, max_degree, n_val=10000, tolerance=1e-7, dps=50):
    """Approximate a function on [-1,1] by Chebyshev polynomials.

    Automatically selects the smallest degree such that the maximum absolute
    error on a validation set does not exceed `tolerance`.

    Args:
        f (callable): Function to approximate. Must accept an mpmath.mpf argument
            and return an mpmath.mpf (or a value convertible to float).
        max_degree (int): Maximum degree to consider.
        n_val (int): Number of validation points (default 10000).
        tolerance (float): Allowed maximum absolute error (default 1e-7).
        dps (int): Decimal precision for mpmath calculations (default 50).

    Returns:
        tuple:
            - approx_func (callable): Approximation function that takes a float and returns a float.
            - coeffs (np.ndarray): Standard polynomial coefficients from lowest to highest power.
            - x_test (np.ndarray): Validation points used.
            - errors (np.ndarray): Absolute errors at validation points for the selected degree.
    """
    mp.dps = dps
    x_test = _generate_validation_points(n_val)

    # Compute Chebyshev coefficients up to max_degree
    full_coeffs_cheb = [mp.mpf(0) for _ in range(max_degree + 1)]
    for k in range(max_degree + 1):
        integrand = lambda theta, k=k: f(mp.cos(theta)) * mp.cos(k * theta)
        I = mp.quad(integrand, [0, mp.pi])
        full_coeffs_cheb[k] = (2 / mp.pi) * I

    best_deg, best_coeffs, best_errors, _ = _find_best_degree(
        max_degree, full_coeffs_cheb, x_test, f, tolerance
    )

    def approx_func(x):
        return _eval_poly(best_coeffs, x)

    return approx_func, best_coeffs, x_test, best_errors


def chebyshev_approximation_even(f, max_degree, n_val=10000, tolerance=1e-7, dps=50):
    """Approximate an even function on [-1,1] by Chebyshev polynomials.

    Uses the evenness property: only even-degree Chebyshev coefficients are non-zero.
    The integration is performed on [0, π/2] with factor 4/π.

    Args:
        f (callable): Even function to approximate. Must accept an mpmath.mpf argument
            and return an mpmath.mpf.
        max_degree (int): Maximum degree to consider.
        n_val (int): Number of validation points (default 10000).
        tolerance (float): Allowed maximum absolute error (default 1e-7).
        dps (int): Decimal precision for mpmath calculations (default 50).

    Returns:
        tuple: Same as `chebyshev_approximation`.
    """
    mp.dps = dps
    x_test = _generate_validation_points(n_val)

    # Compute only even Chebyshev coefficients
    full_coeffs_cheb = [mp.mpf(0) for _ in range(max_degree + 1)]
    for k in range(0, max_degree + 1, 2):
        integrand = lambda theta, k=k: f(mp.cos(theta)) * mp.cos(k * theta)
        I = mp.quad(integrand, [0, mp.pi / 2])
        full_coeffs_cheb[k] = (4 / mp.pi) * I

    best_deg, best_coeffs, best_errors, _ = _find_best_degree(
        max_degree, full_coeffs_cheb, x_test, f, tolerance
    )

    def approx_func(x):
        return _eval_poly(best_coeffs, x)

    return approx_func, best_coeffs, x_test, best_errors


def chebyshev_approximation_odd(f, max_degree, n_val=10000, tolerance=1e-7, dps=50):
    """Approximate an odd function on [-1,1] by Chebyshev polynomials.

    Uses the oddness property: only odd-degree Chebyshev coefficients are non-zero.
    The integration is performed on [0, π/2] with factor 4/π.

    Args:
        f (callable): Odd function to approximate. Must accept an mpmath.mpf argument
            and return an mpmath.mpf.
        max_degree (int): Maximum degree to consider.
        n_val (int): Number of validation points (default 10000).
        tolerance (float): Allowed maximum absolute error (default 1e-7).
        dps (int): Decimal precision for mpmath calculations (default 50).

    Returns:
        tuple: Same as `chebyshev_approximation`.
    """
    mp.dps = dps
    x_test = _generate_validation_points(n_val)

    # Compute only odd Chebyshev coefficients
    full_coeffs_cheb = [mp.mpf(0) for _ in range(max_degree + 1)]
    for k in range(1, max_degree + 1, 2):
        integrand = lambda theta, k=k: f(mp.cos(theta)) * mp.cos(k * theta)
        I = mp.quad(integrand, [0, mp.pi / 2])
        full_coeffs_cheb[k] = (4 / mp.pi) * I

    best_deg, best_coeffs, best_errors, _ = _find_best_degree(
        max_degree, full_coeffs_cheb, x_test, f, tolerance
    )

    def approx_func(x):
        return _eval_poly(best_coeffs, x)

    return approx_func, best_coeffs, x_test, best_errors


# ----------------------------------------------------------------------
# Utility functions for printing
# ----------------------------------------------------------------------

def print_poly(coeffs, var='x', precision=10, dps=50):
    """Print polynomial in standard power form.

    Args:
        coeffs (array-like): Coefficients from lowest to highest power.
        var (str): Variable name (default 'x').
        precision (int): Number of significant digits to display (default 10).
    """
    terms = []
    for i, c in enumerate(coeffs):
        if abs(c) > 10.**(-dps):
            c_str = f"{c:.{precision}e}"
            if i == 0:
                terms.append(c_str)
            elif i == 1:
                terms.append(f"{c_str}*{var}")
            else:
                terms.append(f"{c_str}*{var}^{i}")
    print(" + ".join(terms) if terms else "0")


def print_cheb(coeffs, precision=10, dps=50):
    """Print polynomial in Chebyshev basis: a0/2 + a1*T1(x) + a2*T2(x) + ...

    Args:
        coeffs (array-like): Coefficients in order [a0/2, a1, a2, ..., an].
        precision (int): Number of significant digits to display (default 10).
    """
    terms = []
    if abs(coeffs[0]) > 10.**(-dps):
        terms.append(f"{coeffs[0]:.{precision}e}")
    for k, c in enumerate(coeffs[1:], start=1):
        if abs(c) > 10.**(-dps):
            c_str = f"{c:.{precision}g}"
            terms.append(f"{c_str}*T_{k}(x)")
    print(" + ".join(terms) if terms else "0")


# ----------------------------------------------------------------------
# Example usage (when run as main script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example 1: exponential function exp(x)
    def f_exp(x):
        return mp.exp(x)

    approx, coeffs, x_test, errors = chebyshev_approximation(
        f_exp, max_degree=10, n_val=1000, tolerance=1e-20
    )
    print("=== exp(x) ===")
    print(f"Selected degree: {len(coeffs)-1}")
    print("Polynomial coefficients (lowest to highest power):")
    print_poly(coeffs)
    print(f"Maximum error: {np.max(errors):.2e}")
    plt.plot(x_test, errors)
    plt.show()



    # Example 2: even function I0(beta*sqrt(1-x^2))/I0(beta)
    beta = mp.mpf(8.0)
    I0_beta = mp.besseli(0, beta)

    def f_even(x):
        arg = beta * mp.sqrt(1 - x * x)
        return mp.besseli(0, arg) / I0_beta

    approx_even, coeffs_even, x_test_even, errors_even = chebyshev_approximation_even(
        f_even, max_degree=20, tolerance=1e-9
    )
    print("\n=== Even function (I0) ===")
    print(f"Selected degree: {len(coeffs_even)-1}")
    print("Polynomial coefficients (lowest to highest power):")
    print_poly(coeffs_even)
    print(f"Maximum error: {np.max(errors_even):.2e}")
    plt.plot(x_test_even, errors_even)
    plt.show()

    # Example 3: odd function sin(pi*x)/pi
    def f_odd(x):
        return mp.sin(mp.pi * x) / mp.pi

    approx_odd, coeffs_odd, x_test_odd, errors_odd = chebyshev_approximation_odd(
        f_odd, max_degree=16, tolerance=1e-5
    )
    print("\n=== Odd function sin(pi*x)/pi ===")
    print(f"Selected degree: {len(coeffs_odd)-1}")
    print("Polynomial coefficients (lowest to highest power):")
    print_poly(coeffs_odd)
    print(f"Maximum error: {np.max(errors_odd):.2e}")
    plt.plot(x_test_odd, errors_odd)
    plt.show()
