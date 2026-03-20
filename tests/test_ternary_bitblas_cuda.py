import pytest
import torch

from multiplier.bit_1_58.pytorch import PytorchMultiplier

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

try:
    from multiplier.bit_1_58.cuda.bitblas import BitBLASTernaryMultiplier
except Exception:  # pragma: no cover - import-time skip path
    BitBLASTernaryMultiplier = None


def random_ternary_matrix(n, device):
    return torch.randint(-1, 2, (n, n), dtype=torch.float32, device=device)


def random_vector(n, device):
    return torch.randn(n, dtype=torch.float32, device=device)


def make_bitblas_multiplier(M):
    if BitBLASTernaryMultiplier is None:
        pytest.skip("official BitBLAS package/baseline is unavailable")
    try:
        return BitBLASTernaryMultiplier(M)
    except RuntimeError as e:
        msg = str(e)
        if (
            "No optimized function available" in msg
            or "Unsupported architecture" in msg
            or "unavailable for shape" in msg
        ):
            pytest.skip(msg)
        raise


@pytest.fixture(scope="module")
def n():
    return 1024


class TestTernaryBitBLASCudaMatchesPytorch:
    def test_random_matrix_random_vector(self, n):
        torch.manual_seed(0)
        M = random_ternary_matrix(n, "cuda")
        v = random_vector(n, "cuda")

        expected = PytorchMultiplier(M)(v)
        actual = make_bitblas_multiplier(M)(v).float()

        torch.testing.assert_close(actual, expected, atol=1.5, rtol=0.08)

    def test_all_zeros_matrix(self, n):
        M = torch.zeros(n, n, dtype=torch.float32, device="cuda")
        v = random_vector(n, "cuda")

        actual = make_bitblas_multiplier(M)(v).float()
        torch.testing.assert_close(actual, torch.zeros(n, device="cuda"), atol=1e-5, rtol=0)

    def test_multiple_vectors_same_matrix(self, n):
        torch.manual_seed(1)
        M = random_ternary_matrix(n, "cuda")
        bitblas = make_bitblas_multiplier(M)
        pytorch = PytorchMultiplier(M)

        for _ in range(3):
            v = random_vector(n, "cuda")
            torch.testing.assert_close(bitblas(v).float(), pytorch(v), atol=1.5, rtol=0.08)
