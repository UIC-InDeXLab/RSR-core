"""Tests for ternary RSR multipliers (bit_1_58)."""

import torch
import pytest

from multiplier.bit_1_58.pytorch import PytorchMultiplier
from multiplier.bit_1_58.rsr_py import RSRTernaryV1_0Multiplier
from multiplier.bit_1_58.cpu.rsr_v1_1 import RSRTernaryV1_1Multiplier
from multiplier.bit_1_58.cpu.rsr_v1_2 import RSRTernaryV1_2Multiplier
from multiplier.bit_1_58.cpu.rsr_v1_3 import RSRTernaryV1_3Multiplier
from multiplier.bit_1_58.cpu.rsr_v1_4 import RSRTernaryV1_4Multiplier
from multiplier.bit_1_58.cpu.rsr_v1_5 import RSRTernaryV1_5Multiplier
from multiplier.bit_1_58.cpu.rsr_v1_6 import RSRTernaryV1_6Multiplier
from multiplier.bit_1_58.cpu.rsr_adaptive import RSRTernaryAdaptiveMultiplier
from multiplier.bit_1_58.cpu.bitnet import BitNetTernaryMultiplier
from multiplier.bit_1_58.cpu.tmac import TMACTernaryMultiplier


@pytest.fixture(params=[16, 32, 64])
def n(request):
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def k(request):
    return request.param


def random_ternary_matrix(n):
    """Generate a random ternary matrix with values in {-1, 0, +1}."""
    return torch.randint(-1, 2, (n, n), dtype=torch.float32)


def random_vector(n):
    return torch.randn(n, dtype=torch.float32)


# ---------------------------------------------------------------------------
# v1.0 — two separate binary RSRs (pure PyTorch)
# ---------------------------------------------------------------------------

class TestTernaryRSRV1_0:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV1_0Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_0Multiplier(M, k)(v)
        torch.testing.assert_close(actual, v)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_0Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_all_zeros(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.zeros(n, n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_0Multiplier(M, k)(v)
        torch.testing.assert_close(actual, torch.zeros(n))

    def test_all_ones(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.ones(n, n, dtype=torch.float32)
        v = random_vector(n)
        expected = torch.full((n,), v.sum().item())
        actual = RSRTernaryV1_0Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_all_neg_ones(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.ones(n, n, dtype=torch.float32)
        v = random_vector(n)
        expected = torch.full((n,), -v.sum().item())
        actual = RSRTernaryV1_0Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV1_0Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


# ---------------------------------------------------------------------------
# v1.1 — fused block loop
# ---------------------------------------------------------------------------

class TestTernaryRSRV1_1:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV1_1Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_1Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV1_1Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


# ---------------------------------------------------------------------------
# v1.2 — single ternary encoding (2k-bit)
# ---------------------------------------------------------------------------

class TestTernaryRSRV1_2:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV1_2Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_2Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV1_2Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


# ---------------------------------------------------------------------------
# v1.3 — batched across blocks
# ---------------------------------------------------------------------------

class TestTernaryRSRV1_3:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV1_3Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_3Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV1_3Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


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
# v1.5 — C kernel with AVX2
# ---------------------------------------------------------------------------

class TestTernaryRSRV1_5:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV1_5Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_5Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_all_zeros(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.zeros(n, n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_5Multiplier(M, k)(v)
        torch.testing.assert_close(actual, torch.zeros(n))

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV1_5Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


# ---------------------------------------------------------------------------
# v1.6 — C kernel with AVX2, two-pass permute + linear scan
# ---------------------------------------------------------------------------

class TestTernaryRSRV1_6:
    def test_random(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryV1_6Multiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    def test_neg_identity(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_6Multiplier(M, k)(v)
        torch.testing.assert_close(actual, -v)

    def test_all_zeros(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.zeros(n, n, dtype=torch.float32)
        v = random_vector(n)
        actual = RSRTernaryV1_6Multiplier(M, k)(v)
        torch.testing.assert_close(actual, torch.zeros(n))

    def test_multiple_vectors(self, n, k):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n)
        rsr = RSRTernaryV1_6Multiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


# ---------------------------------------------------------------------------
# Adaptive — any k via zero-padding
# ---------------------------------------------------------------------------

class TestTernaryRSRAdaptive:
    @pytest.mark.parametrize("n,k", [(15, 4), (17, 6), (31, 8), (32, 8), (33, 12)])
    def test_random(self, n, k):
        M = random_ternary_matrix(n)
        v = random_vector(n)
        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryAdaptiveMultiplier(M, k)(v)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("n,k", [(15, 4), (17, 6), (33, 12)])
    def test_multiple_vectors(self, n, k):
        M = random_ternary_matrix(n)
        rsr = RSRTernaryAdaptiveMultiplier(M, k)
        pytorch = PytorchMultiplier(M)
        for _ in range(5):
            v = random_vector(n)
            torch.testing.assert_close(rsr(v), pytorch(v))


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
