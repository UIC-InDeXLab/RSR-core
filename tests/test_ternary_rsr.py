"""Tests for ternary RSR multipliers (bit_1_58)."""

import torch
import pytest

from multiplier.bit_1_58.pytorch import PytorchMultiplier
from multiplier.bit_1_58.cpu.rsr_v1_4 import RSRTernaryV1_4Multiplier
from multiplier.bit_1_58.cpu.rsr_v3_1 import RSRTernaryV3_1Multiplier
from multiplier.bit_1_58.cpu.rsr_v3_3 import RSRTernaryV3_3Multiplier
from multiplier.bit_1_58.cpu.rsr_nonsquare import RSRTernaryNonSquareMultiplier
from multiplier.bit_1_58.cpu.bitnet import BitNetTernaryMultiplier
from multiplier.bit_1_58.cpu.tmac import TMACTernaryMultiplier


@pytest.fixture(params=[16, 32, 64])
def n(request):
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def k(request):
    return request.param


def random_ternary_matrix(n, m=None):
    """Generate a random ternary matrix with values in {-1, 0, +1}."""
    if m is None:
        m = n
    return torch.randint(-1, 2, (n, m), dtype=torch.float32)


def random_vector(n):
    return torch.randn(n, dtype=torch.float32)


# ---------------------------------------------------------------------------
# v1.4 — C kernel (fused ternary)
# ---------------------------------------------------------------------------

class TestTernaryRSRV1_4:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV1_4Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_4Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_all_zeros(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.zeros(n, n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_4Multiplier(M, k)(v)
        torch.testing.assert_close(actual, torch.zeros(n))

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV1_4Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


# ---------------------------------------------------------------------------
# v3.1 — direct kernel with 16-bit metadata
# ---------------------------------------------------------------------------

class TestTernaryRSRV3_1:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV3_1Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV3_1Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV3_1Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


# ---------------------------------------------------------------------------
# v3.3 — direct kernel with scatter bitmasks
# ---------------------------------------------------------------------------

class TestTernaryRSRV3_3:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV3_3Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV3_3Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV3_3Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


# ---------------------------------------------------------------------------
# Non-square — unified multiplier with v3.1/v3.3 switching
# ---------------------------------------------------------------------------

class TestTernaryRSRNonSquare:
    @pytest.mark.parametrize("n_rows,n_cols,k", [
        (16, 32, 2), (32, 16, 4), (64, 64, 8),
        (15, 32, 4), (33, 64, 8),  # padding cases
    ])
    def test_nonsquare_random(self, n_rows, n_cols, k):
        M = random_ternary_matrix(n_rows, n_cols)
        v = random_vector(n_cols)
        expected = M @ v
        actual = RSRTernaryNonSquareMultiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_square(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryNonSquareMultiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryNonSquareMultiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)


# ---------------------------------------------------------------------------
# BitNet ternary baseline (official I2_S — int8 activation quantization)
# ---------------------------------------------------------------------------

class TestBitNetTernary:
    """BitNet I2_S uses int8 quantization so results are approximate."""

    def test_random(self, n):
        if n % 4 != 0:
            pytest.skip(f"n={n} not divisible by 4")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = BitNetTernaryMultiplier(M)(v)
        torch.testing.assert_close(actual, expected, atol=0.15, rtol=0.05)

    def test_identity(self, n):
        if n % 4 != 0:
            pytest.skip(f"n={n} not divisible by 4")
        M = torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = BitNetTernaryMultiplier(M)(v)
        torch.testing.assert_close(actual, v, atol=0.15, rtol=0.05)

    def test_neg_identity(self, n):
        if n % 4 != 0:
            pytest.skip(f"n={n} not divisible by 4")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = BitNetTernaryMultiplier(M)(v)
        torch.testing.assert_close(actual, -v, atol=0.15, rtol=0.05)

    def test_all_zeros(self, n):
        if n % 4 != 0:
            pytest.skip(f"n={n} not divisible by 4")
        M = torch.zeros(n, n, dtype=torch.float32)
        v = random_vector(n)
        actual = BitNetTernaryMultiplier(M)(v)
        torch.testing.assert_close(actual, torch.zeros(n), atol=1e-6, rtol=0)

    def test_multiple_vectors(self, n):
        if n % 4 != 0:
            pytest.skip(f"n={n} not divisible by 4")
        M = random_ternary_matrix(n)
        bn = BitNetTernaryMultiplier(M)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(bn(v), pytorch(v), atol=0.15, rtol=0.05)


# ---------------------------------------------------------------------------
# T-MAC ternary baseline (LUT-based — int8 LUT quantization)
# ---------------------------------------------------------------------------

class TestTMACTernary:
    """T-MAC uses int8 LUT quantization so results are approximate."""

    def test_random(self, n):
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = TMACTernaryMultiplier(M)(v)
        torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.05)

    def test_all_zeros(self, n):
        M = torch.zeros(n, n, dtype=torch.float32)
        v = random_vector(n)
        actual = TMACTernaryMultiplier(M)(v)
        torch.testing.assert_close(actual, torch.zeros(n), atol=1e-6, rtol=0)

    def test_neg_identity(self, n):
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = TMACTernaryMultiplier(M)(v)
        torch.testing.assert_close(actual, -v, atol=0.2, rtol=0.05)

    def test_multiple_vectors(self, n):
        M = random_ternary_matrix(n)
        tmac = TMACTernaryMultiplier(M)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(tmac(v), pytorch(v), atol=0.2, rtol=0.05)
