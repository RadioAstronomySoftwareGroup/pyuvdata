# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Define some commonly used numpy array type hint aliases."""

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]
StrArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

Int64Array = NDArray[np.int64]
UInt64Array = NDArray[np.uint64]
