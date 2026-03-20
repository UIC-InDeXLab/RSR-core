import pytest
import torch

from multiplier.bit_1_58.pytorch import PytorchMultiplier

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

try:
    from multiplier.bit_1_58.cuda.bitnet import BitNetCudaOfficialMultiplier
except Exception as _BITNET_IMPORT_ERROR:  # pragma: no cover - import-time skip path
    BitNetCudaOfficialMultiplier = None


def random_ternary_matrix(n, device):
    return torch.randint(-1, 2, (n, n), dtype=torch.float32, device=device)


def random_vector(n, device):
    return torch.randn(n, dtype=torch.float32, device=device)


@pytest.fixture(scope="module")
def n():
    return 1024


class TestTernaryBitNetCudaMatchesPytorch:
    def test_random_matrix_random_vector(self, n):
        if BitNetCudaOfficialMultiplier is None:
            pytest.skip("official BitNet CUDA kernel failed to import/build")

        torch.manual_seed(0)
        M = random_ternary_matrix(n, "cuda")
        v = random_vector(n, "cuda")

        expected = PytorchMultiplier(M)(v)
        actual = BitNetCudaOfficialMultiplier(M)(v).float()

        torch.testing.assert_close(actual, expected, atol=1.5, rtol=0.08)

    def test_all_zeros_matrix(self, n):
        if BitNetCudaOfficialMultiplier is None:
            pytest.skip("official BitNet CUDA kernel failed to import/build")

        M = torch.zeros(n, n, dtype=torch.float32, device="cuda")
        v = random_vector(n, "cuda")

        actual = BitNetCudaOfficialMultiplier(M)(v).float()
        torch.testing.assert_close(actual, torch.zeros(n, device="cuda"), atol=1e-5, rtol=0)

    def test_multiple_vectors_same_matrix(self, n):
        if BitNetCudaOfficialMultiplier is None:
            pytest.skip("official BitNet CUDA kernel failed to import/build")

        torch.manual_seed(1)
        M = random_ternary_matrix(n, "cuda")
        bitnet = BitNetCudaOfficialMultiplier(M)
        pytorch = PytorchMultiplier(M)

        for _ in range(3):
            v = random_vector(n, "cuda")
            torch.testing.assert_close(bitnet(v).float(), pytorch(v), atol=1.5, rtol=0.08)
