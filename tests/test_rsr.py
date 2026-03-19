import importlib
import torch
import pytest

from multiplier.bit_1.pytorch import PytorchMultiplier
from multiplier.bit_1.cpu.bitnet import BitNetOfficialMultiplier
from multiplier.bit_1.cpu.tmac import TMACBinaryMultiplier
from multiplier.bit_1.rsr_py import RSRPythonMultiplier
from multiplier.bit_1.cpu.rsr_cpp import RSRCppMultiplier
from multiplier.bit_1.cpu.rsr_adaptive import RSRAdaptiveMultiplier

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=[16, 32, 64])
def n(request):
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def k(request):
    return request.param


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def random_binary_matrix(n, device):
    return torch.randint(0, 2, (n, n), dtype=torch.float32, device=device)


def random_vector(n, device):
    return torch.randn(n, dtype=torch.float32, device=device)


class TestRSRMatchesPytorch:
    def test_random_matrix_random_vector(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_binary_matrix(n, device)
        v = random_vector(n, device)

        expected = PytorchMultiplier(M)(v)
        actual = RSRPythonMultiplier(M, k)(v)

        torch.testing.assert_close(actual, expected)

    def test_identity_matrix(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.eye(n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        actual = RSRPythonMultiplier(M, k)(v)

        torch.testing.assert_close(actual, v)

    def test_all_ones_matrix(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.ones(n, n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        expected = torch.full((n,), v.sum().item(), device=device)
        actual = RSRPythonMultiplier(M, k)(v)

        torch.testing.assert_close(actual, expected)

    def test_all_zeros_matrix(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.zeros(n, n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        actual = RSRPythonMultiplier(M, k)(v)

        torch.testing.assert_close(actual, torch.zeros(n, device=device))

    def test_binary_vector(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_binary_matrix(n, device)
        v = torch.randint(0, 2, (n,), dtype=torch.float32, device=device)

        expected = PytorchMultiplier(M)(v)
        actual = RSRPythonMultiplier(M, k)(v)

        torch.testing.assert_close(actual, expected)

    def test_multiple_vectors_same_matrix(self, n, k, device):
        """Preprocessing is done once; verify multiple inferences are correct."""
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_binary_matrix(n, device)
        rsr = RSRPythonMultiplier(M, k)
        pytorch = PytorchMultiplier(M)

        for _ in range(5):
            v = random_vector(n, device)
            torch.testing.assert_close(rsr(v), pytorch(v))


class TestRSRCppMatchesPytorch:
    def test_random_matrix_random_vector(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_binary_matrix(n, device)
        v = random_vector(n, device)

        expected = PytorchMultiplier(M)(v)
        actual = RSRCppMultiplier(M, k)(v)

        torch.testing.assert_close(actual, expected)

    def test_identity_matrix(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.eye(n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        actual = RSRCppMultiplier(M, k)(v)

        torch.testing.assert_close(actual, v)

    def test_all_ones_matrix(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.ones(n, n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        expected = torch.full((n,), v.sum().item(), device=device)
        actual = RSRCppMultiplier(M, k)(v)

        torch.testing.assert_close(actual, expected)

    def test_all_zeros_matrix(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = torch.zeros(n, n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        actual = RSRCppMultiplier(M, k)(v)

        torch.testing.assert_close(actual, torch.zeros(n, device=device))

    def test_multiple_vectors_same_matrix(self, n, k, device):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_binary_matrix(n, device)
        rsr = RSRCppMultiplier(M, k)
        pytorch = PytorchMultiplier(M)

        for _ in range(5):
            v = random_vector(n, device)
            torch.testing.assert_close(rsr(v), pytorch(v))


class TestRSRAdaptiveMatchesPytorch:
    @pytest.mark.parametrize("n,k", [(15, 4), (17, 6), (31, 8), (32, 8), (33, 16)])
    def test_random_matrix_random_vector_any_k(self, n, k, device):
        M = random_binary_matrix(n, device)
        v = random_vector(n, device)

        expected = PytorchMultiplier(M)(v)
        actual = RSRAdaptiveMultiplier(M, k)(v)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("n,k", [(15, 4), (17, 6), (33, 16)])
    def test_multiple_vectors_same_matrix(self, n, k, device):
        M = random_binary_matrix(n, device)
        rsr = RSRAdaptiveMultiplier(M, k)
        pytorch = PytorchMultiplier(M)

        for _ in range(5):
            v = random_vector(n, device)
            torch.testing.assert_close(rsr(v), pytorch(v))


class TestBitNetOfficialMatchesPytorch:
    """Official-style I2_S baseline still quantizes activations, so approximate."""

    def test_random_matrix_random_vector(self, n, device):
        if n % 4 != 0:
            pytest.skip(f"n={n} not divisible by 4")
        M = random_binary_matrix(n, device)
        v = random_vector(n, device)

        expected = PytorchMultiplier(M)(v)
        actual = BitNetOfficialMultiplier(M)(v)

        torch.testing.assert_close(actual, expected, atol=0.15, rtol=0.05)

    def test_all_zeros_matrix(self, n, device):
        if n % 4 != 0:
            pytest.skip(f"n={n} not divisible by 4")
        M = torch.zeros(n, n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        actual = BitNetOfficialMultiplier(M)(v)
        torch.testing.assert_close(actual, torch.zeros(n, device=device), atol=1e-6, rtol=0)

    def test_multiple_vectors_same_matrix(self, n, device):
        if n % 4 != 0:
            pytest.skip(f"n={n} not divisible by 4")
        M = random_binary_matrix(n, device)
        bitnet = BitNetOfficialMultiplier(M)
        pytorch = PytorchMultiplier(M)

        for _ in range(5):
            v = random_vector(n, device)
            torch.testing.assert_close(bitnet(v), pytorch(v), atol=0.15, rtol=0.05)


class TestTMACBinaryMatchesPytorch:
    """T-MAC uses int8 LUT quantization so results are approximate."""

    def test_random_matrix_random_vector(self, n, device):
        M = random_binary_matrix(n, device)
        v = random_vector(n, device)

        expected = PytorchMultiplier(M)(v)
        actual = TMACBinaryMultiplier(M)(v)

        torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.05)

    def test_all_zeros_matrix(self, n, device):
        M = torch.zeros(n, n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        actual = TMACBinaryMultiplier(M)(v)

        torch.testing.assert_close(actual, torch.zeros(n, device=device), atol=0.05, rtol=0)

    def test_identity_matrix(self, n, device):
        M = torch.eye(n, dtype=torch.float32, device=device)
        v = random_vector(n, device)

        actual = TMACBinaryMultiplier(M)(v)

        torch.testing.assert_close(actual, v, atol=0.2, rtol=0.05)

    def test_multiple_vectors_same_matrix(self, n, device):
        M = random_binary_matrix(n, device)
        tmac = TMACBinaryMultiplier(M)
        pytorch = PytorchMultiplier(M)

        for _ in range(5):
            v = random_vector(n, device)
            torch.testing.assert_close(tmac(v), pytorch(v), atol=0.2, rtol=0.05)
