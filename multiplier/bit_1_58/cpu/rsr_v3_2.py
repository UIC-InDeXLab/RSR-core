"""Ternary RSR v3.2: runtime hybrid between v3.0 and v3.1."""

import numpy as np

from ._rsr_v3_common import INT8_PTR, INT32_PTR, UINT16_PTR
from .rsr_v1_4 import RSRTernaryV1_4Multiplier
from .rsr_v3_0 import RSRTernaryV3_0Multiplier
from .rsr_v3_1 import RSRTernaryV3_1Multiplier


class RSRTernaryV3_2Multiplier(RSRTernaryV1_4Multiplier):
    """Dispatch between the two-pass and direct kernels from group statistics."""

    def prep(self):
        super().prep()

        self._perms_ptr = self._perms.ctypes.data_as(INT32_PTR)
        self._group_ends_ptr = self._group_ends.ctypes.data_as(INT32_PTR)
        self._scatter_offsets_ptr = self._scatter_offsets.ctypes.data_as(INT32_PTR)
        self._scatter_rows_ptr = self._scatter_rows.ctypes.data_as(INT8_PTR)
        self._scatter_signs_ptr = self._scatter_signs.ctypes.data_as(INT8_PTR)
        self._block_meta_ptr = self._block_meta.ctypes.data_as(INT32_PTR)

        assert self.n <= np.iinfo(np.uint16).max, "v3.2 requires n <= 65535"
        self._perms_u16 = self._perms.astype(np.uint16, copy=True)
        self._group_ends_u16 = self._group_ends.astype(np.uint16, copy=True)
        self._perms_u16_ptr = self._perms_u16.ctypes.data_as(UINT16_PTR)
        self._group_ends_u16_ptr = self._group_ends_u16.ctypes.data_as(UINT16_PTR)

        total_groups = len(self._group_ends)
        avg_group_len = (self.n * self._num_blocks) / max(total_groups, 1)
        self._use_two_pass = avg_group_len >= 48.0

    def __call__(self, v):
        if self._use_two_pass:
            return RSRTernaryV3_0Multiplier.__call__(self, v)
        return RSRTernaryV3_1Multiplier.__call__(self, v)
