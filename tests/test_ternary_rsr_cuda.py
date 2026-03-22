import pytest
import torch

from multiplier.bit_1_58.pytorch import PytorchMultiplier
from multiplier.bit_1_58.cuda.rsr_cuda_v6_0 import RSRTernaryCudaV6_0Multiplier
from multiplier.bit_1_58.cuda.rsr_cuda_v6_3 import RSRTernaryCudaV6_3Multiplier
from multiplier.bit_1_58.cuda.rsr_cuda_v6_5 import RSRTernaryCudaV6_5Multiplier
from multiplier.bit_1_58.cuda.rsr_cuda_v10_1 import RSRTernaryCudaV10_1Multiplier
from multiplier.bit_1_58.cuda.rsr_cuda_adaptive import RSRTernaryCudaAdaptiveMultiplier

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(params=[16, 32, 64])
def n(request):
    return request.param


@pytest.fixture(params=[2, 4, 8])
def k(request):
    return request.param


def random_ternary_matrix(n, device):
    return torch.randint(-1, 2, (n, n), dtype=torch.float32, device=device)


def random_vector(n, device):
    return torch.randn(n, dtype=torch.float32, device=device)


@pytest.mark.parametrize(
    "cls",
    [
        RSRTernaryCudaV6_0Multiplier,
        RSRTernaryCudaV6_3Multiplier,
        RSRTernaryCudaV6_5Multiplier,
        RSRTernaryCudaV10_1Multiplier,
    ],
)
class TestTernaryCudaMatchesPytorch:
    def test_random_matrix_random_vector(self, n, k, cls):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = random_ternary_matrix(n, "cuda")
        v = random_vector(n, "cuda")

        expected = PytorchMultiplier(M)(v)
        actual = cls(M, k)(v)
        tol = 0.02 * n**0.5
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    def test_neg_identity(self, n, k, cls):
        if n % k != 0:
            pytest.skip(f"n={n} not divisible by k={k}")
        M = -torch.eye(n, dtype=torch.float32, device="cuda")
        v = random_vector(n, "cuda")

        actual = cls(M, k)(v)
        tol = 0.02 * n**0.5
        torch.testing.assert_close(actual, -v, atol=tol, rtol=tol)


class TestTernaryCudaAdaptiveMatchesPytorch:
    @pytest.mark.parametrize("n,k", [(15, 4), (17, 6), (31, 8), (32, 8), (33, 16)])
    def test_random_matrix_random_vector_any_k(self, n, k):
        M = random_ternary_matrix(n, "cuda")
        v = random_vector(n, "cuda")

        expected = PytorchMultiplier(M)(v)
        actual = RSRTernaryCudaAdaptiveMultiplier(M, k)(v)
        tol = 0.02 * n**0.5
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
