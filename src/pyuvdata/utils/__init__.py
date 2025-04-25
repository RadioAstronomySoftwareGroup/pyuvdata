# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Commonly used utility functions."""

from __future__ import annotations

import numpy as np

# standard angle tolerance: 1 mas in radians.
RADIAN_TOL = 1 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)
# standard lst time tolerance: 5 ms (75 mas in radians), based on an expected RMS
# accuracy of 1 ms at 7 days out from issuance of Bulletin A (which are issued once a
# week with rapidly determined parameters and forecasted values of DUT1), the exact
# formula for which is t_err = 0.00025 (MJD-<Bulletin A Release Data>)**0.75 (in secs).
LST_RAD_TOL = 2 * np.pi * 5e-3 / (86400.0)

# these seem to be necessary for the installed package to access these submodules
from . import antenna  # noqa
from . import apply_uvflag  # noqa
from . import array_collapse  # noqa
from . import bls  # noqa
from . import bltaxis  # noqa
from . import coordinates  # noqa
from . import frequency  # noqa
from . import history  # noqa
from . import io  # noqa
from . import phase_center_catalog  # noqa
from . import phasing  # noqa
from . import pol  # noqa
from . import redundancy  # noqa
from . import times  # noqa
from . import tools  # noqa
from . import uvcalibrate  # noqa

# Add things to the utils namespace used by outside packages
from .apply_uvflag import apply_uvflag  # noqa
from .array_collapse import collapse  # noqa
from .bls import *  # noqa
from .coordinates import *  # noqa
from .phasing import uvw_track_generator  # noqa
from .pol import *  # noqa
from .times import get_lst_for_time  # noqa
from .uvcalibrate import uvcalibrate  # noqa
