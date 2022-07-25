import pytest
import numpy as np
from matrix_pencil import MatrixPencil


def sine_function(t, alpha, amplitude, freq, phase, offset):
    return offset + amplitude * np.exp(alpha * t) * np.sin(freq * t + phase)


def check_sin_no_offset(mp: MatrixPencil, t, y, alpha, amplitude, freq, phase):
    mp_damp, mp_freq = mp.compute(t, y)

    model_order = mp.model_order
    assert model_order == 2
    assert len(mp_freq) == model_order
    sorted_freq = np.sort(mp_freq)
    assert pytest.approx(sorted_freq[0], 1e-12) == -freq
    assert pytest.approx(sorted_freq[1], 1e-12) == freq

    amps = mp.amplitudes
    mp_phase = mp.phases
    for i in range(model_order):
        assert pytest.approx(mp_damp[i], 1e-12) == -alpha / freq
        assert pytest.approx(amps[i], 1e-12) == amplitude / 2.0
        if mp_freq[i] < 0.0:
            assert pytest.approx(mp_phase[i], 1e-12) == -phase + np.pi / 2.0
        else:
            assert pytest.approx(mp_phase[i], 1e-12) == phase - np.pi / 2.0


def test_basic_sin():
    # matrix pencil settings
    mp = MatrixPencil(num_subsamples=-1, output_level=0, svd_tolerance=0.1)

    # signal set up
    n = 1000
    offset = 0.0
    amplitude = 1.0
    alpha = 0.0
    freq = 1.0
    phase = 0.0

    t = np.linspace(0.0, 2.0, n)
    y = sine_function(t, alpha, amplitude, freq, phase, offset)
    check_sin_no_offset(mp, t, y, alpha, amplitude, freq, phase)


def test_basic_sin_with_positive_alpha():
    # matrix pencil settings
    mp = MatrixPencil(num_subsamples=-1, output_level=0, svd_tolerance=0.1)

    # signal set up
    n = 1000
    offset = 0.0
    amplitude = 1.0
    alpha = 0.1
    freq = 1.0
    phase = 0.0

    t = np.linspace(0.0, 2.0, n)
    y = sine_function(t, alpha, amplitude, freq, phase, offset)
    check_sin_no_offset(mp, t, y, alpha, amplitude, freq, phase)


def test_basic_sin_with_negative_alpha():
    # matrix pencil settings
    mp = MatrixPencil(num_subsamples=-1, output_level=0, svd_tolerance=0.1)

    # signal set up
    n = 1000
    offset = 0.0
    amplitude = 1.0
    alpha = -0.1
    freq = 1.0
    phase = 0.0

    t = np.linspace(0.0, 2.0, n)
    y = sine_function(t, alpha, amplitude, freq, phase, offset)
    check_sin_no_offset(mp, t, y, alpha, amplitude, freq, phase)


def test_sin_plus_positive_offset():
    # matrix pencil settings
    mp = MatrixPencil(num_subsamples=-1, output_level=0, svd_tolerance=0.01)

    # signal set up
    n = 1000
    offset = 1.0
    amplitude = 1.0
    alpha = 0.0
    freq = 1.0
    phase = 0.0

    t = np.linspace(0.0, 5.0, n)
    y = sine_function(t, alpha, amplitude, freq, phase, offset)

    mp_damp, mp_freq = mp.compute(t, y)

    model_order = mp.model_order
    assert model_order == 3
    assert len(mp_freq) == model_order
    sorted_freq = np.sort(mp_freq)
    assert pytest.approx(sorted_freq[0], 1e-12) == -freq
    assert pytest.approx(sorted_freq[1], 1e-12) == 0.0
    assert pytest.approx(sorted_freq[2], 1e-12) == freq

    amps = mp.amplitudes
    phase = mp.phases
    for i in range(model_order):

        if mp_freq[i] < -1e-7:
            assert pytest.approx(phase[i], 1e-12) == np.pi / 2.0
            assert pytest.approx(amps[i], 1e-12) == amplitude / 2.0
            assert pytest.approx(mp_damp[i], 1e-12) == -alpha / freq
        elif mp_freq[i] > 1e-7:
            assert pytest.approx(phase[i], 1e-12) == -np.pi / 2.0
            assert pytest.approx(amps[i], 1e-12) == amplitude / 2.0
            assert pytest.approx(mp_damp[i], 1e-12) == -alpha / freq
        else:
            assert pytest.approx(phase[i], 1e-12) == 0.0
            assert pytest.approx(amps[i], 1e-12) == offset


def test_basic_sin_derivative_of_agg_damping_wrt_signal_prescribed_damping():
    mp = MatrixPencil(num_subsamples=-1, output_level=0, svd_tolerance=0.1)

    # signal set up
    n = 1000
    offset = 0.0
    amplitude = 1.0
    alpha = 0.1
    freq = 1.0
    phase = 0.0

    t = np.linspace(0.0, 2.0, n)
    y = sine_function(t, alpha, amplitude, freq, phase, offset)
    mp.compute(t, y)
    agg = mp.compute_aggregate_damping()

    d_agg_dy = mp.compute_aggregate_damping_derivative()
    d_y_d_alpha = amplitude * np.exp(alpha * t) * np.sin(freq * t + phase) * t
    d_agg_d_alpha = d_agg_dy @ d_y_d_alpha

    assert d_agg_d_alpha == pytest.approx(1.0)
