# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Define some commonly used numpy array type hint aliases."""

import numpy as np

IntArray = np.typing.NDArray[np.integer]
FloatArray = np.typing.NDArray[np.floating]
StrArray = np.typing.NDArray[np.str_]
BoolArray = np.typing.NDArray[np.bool_]

Int64Array = np.typing.NDArray[np.int64]
UInt64Array = np.typing.NDArray[np.uint64]
