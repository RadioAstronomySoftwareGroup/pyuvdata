# Copyright 2018 the HERA Project
# Licensed under the 2-clause BSD License

"""Module for low-level interface to MIRIAD files.

This module extracts some Python code from AIPY used in our MIRIAD I/O
routines. It was copied from AIPY commit
6cb5a70876f33dccdd68d4063b076f8d42d9edae, then reformatted. The only items
used by pyuvdata are ``uv_selector`` and ``UV``.

"""

__all__ = ["uv_selector", "UV"]

import contextlib
import os
import re
import warnings

import numpy as np
from astropy import constants as const, units
from astropy.coordinates import EarthLocation

try:
    from . import _miriad
except ImportError as e:
    raise ImportError(
        "The miriad extension is not built but is required for reading miriad "
        "files. Note that miriad is currently not supported on Windows."
    ) from e


str2pol = {
    "I": 1,  # Stokes Paremeters
    "Q": 2,
    "U": 3,
    "V": 4,
    "rr": -1,  # Circular Polarizations
    "ll": -2,
    "rl": -3,
    "lr": -4,
    "xx": -5,  # Linear Polarizations
    "yy": -6,
    "xy": -7,
    "yx": -8,
}


def bl2ij(bl):
    """
    Convert baseline number to antenna numbers.

    Parameters
    ----------
    bl : int
        baseline number

    Returns
    -------
    int
        first antenna number
    int
        second antenna number
    """
    bl = int(bl)

    if bl > 65536:
        bl -= 65536
        mant = 2048
    else:
        mant = 256

    return (bl // mant - 1, bl % mant - 1)


def ij2bl(i, j):
    """
    Convert antenna numbers to baseline number.

    Parameters
    ----------
    i : int
        first antenna number
    j : int
        second antenna number

    Returns
    -------
    int
        baseline number
    """
    if i > j:
        i, j = j, i

    if j + 1 < 256:
        return 256 * (i + 1) + (j + 1)

    return 2048 * (i + 1) + (j + 1) + 65536


ant_re = r"(\(((-?\d+[xy]?,?)+)\)|-?\d+[xy]?)"
bl_re = f"(^({ant_re}_{ant_re}|{ant_re}),?)"


def parse_ants(ant_str, nants):
    """
    Parse ant string into a list of (baseline, include, pol) tuples.

    Generate list of (baseline, include, pol) tuples based on parsing of the
    string associated with the 'ants' command-line option.

    Parameters
    ----------
    ant_str : str
        string associated with the 'ants' command-line option
    nants : int
        number of antennas

    Returns
    -------
    list of tuples
        list of (baseline, include, pol) tuples
    """
    rv, cnt = [], 0

    while cnt < len(ant_str):
        m = re.search(bl_re, ant_str[cnt:])

        if m is None:
            if ant_str[cnt:].startswith("all"):
                rv = []
            elif ant_str[cnt:].startswith("auto") or ant_str[cnt:].startswith("-cross"):
                rv.append(("auto", 1, -1))
            elif ant_str[cnt:].startswith("cross") or ant_str[cnt:].startswith("-auto"):
                rv.append(("auto", 0, -1))
            else:
                raise ValueError(f'Unparsable ant argument "{ant_str}"')
            c = ant_str[cnt:].find(",")

            if c >= 0:
                cnt += c + 1
            else:
                cnt = len(ant_str)
        else:
            m = m.groups()
            cnt += len(m[0])

            if m[2] is None:
                ais = [m[8]]
                ajs = list(range(nants))
            else:
                if m[3] is None:
                    ais = [m[2]]
                else:
                    ais = m[3].split(",")

                if m[6] is None:
                    ajs = [m[5]]
                else:
                    ajs = m[6].split(",")

            for i in ais:
                if isinstance(i, str) and i.startswith("-"):
                    i = i[1:]  # nibble the - off the string
                    include_i = 0
                else:
                    include_i = 1

                for j in ajs:
                    include = None

                    if isinstance(j, str) and j.startswith("-"):
                        j = j[1:]
                        include_j = 0
                    else:
                        include_j = 1

                    include = int(include_i and include_j)
                    pol = None
                    i, j = str(i), str(j)

                    if not i.isdigit():
                        ai = re.search(r"(\d+)([x,y])", i).groups()
                    if not j.isdigit():
                        aj = re.search(r"(\d+)([x,y])", j).groups()

                    if i.isdigit() and not j.isdigit():
                        pol = ["x" + aj[1], "y" + aj[1]]
                        ai = [i, ""]
                    elif not i.isdigit() and j.isdigit():
                        pol = [ai[1] + "x", ai[1] + "y"]
                        aj = [j, ""]
                    elif not i.isdigit() and not j.isdigit():
                        pol = [ai[1] + aj[1]]

                    if pol is not None:
                        bl = ij2bl(abs(int(ai[0])), abs(int(aj[0])))
                        for p in pol:
                            rv.append((bl, include, p))
                    else:
                        bl = ij2bl(abs(int(i)), abs(int(j)))
                        rv.append((bl, include, -1))
    return rv


def uv_selector(uv, ants=-1, pol_str=-1):
    """
    Call select on a Miriad object with string arguments for antennas and polarizations.

    Parameters
    ----------
    uv : UV object
        Miriad data set object
    ants : str
        string to select antennas or baselines, e.g. 'all', 'auto', 'cross',
        '0,1,2', or '0_1,0_2'
    pol_str : str
        polarizations to select, e.g. 'xx', 'yy', 'xy', 'yx'

    Returns
    -------
    None
    """
    if ants != -1:
        if isinstance(ants, str):
            ants = parse_ants(ants, uv["nants"])

        for cnt, (bl, include, pol) in enumerate(ants):
            if cnt > 0:
                if include:
                    uv.select("or", -1, -1)
                else:
                    uv.select("and", -1, -1)

            if pol == -1:
                pol = pol_str  # default to explicit pol parameter

            if bl == "auto":
                uv.select("auto", 0, 0, include=include)
            else:
                i, j = bl2ij(bl)
                uv.select("antennae", i, j, include=include)

            if pol != -1:
                for p in pol.split(","):
                    polopt = str2pol[p]
                    uv.select("polarization", polopt, 0)
    elif pol_str != -1:
        for p in pol_str.split(","):
            polopt = str2pol[p]
            uv.select("polarization", polopt, 0)


itemtable = {
    "obstype": "a",
    "history": "a",
    "vartable": "a",
    "ngains": "i",
    "nfeeds": "i",
    "ntau": "i",
    "nsols": "i",
    "interval": "d",
    "leakage": "?",
    "gains": "?",
    "freq0": "d",
    "freqs": "?",
    "bandpass": "?",
    "nbpsols": "i",
    "nspect0": "i",
    "nchan0": "i",
    "stopt": "d",
    "duration": "d",
}


def _uv_pipe_default_action(uv, p, d, f=None):
    if f is None:
        return p, d
    else:
        return p, d, f


class UV(_miriad.UV):
    """Top-level interface to a Miriad UV data set."""

    def __init__(self, filename, status="old", corrmode="r"):
        """
        Initialize from a miriad file.

        Parameters
        ----------
        filename : str
            filename to initialize from
        status : str
            options are: 'old', 'new', 'append'
        corrmode : str
            options are 'r' (float32 data storage) or 'j' (int16 with shared exponent)
        """
        if status not in ["old", "new", "append"]:  # pragma: no cover
            raise RuntimeError(
                "Something went wrong in aipy_extracts.__init__. Please "
                "file an issue in our GitHub issue log so that we can help: "
                "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues."
                " Developer info: unknown status"
            )
        if corrmode not in ["r", "j"]:  # pragma: no cover
            raise RuntimeError(
                "Something went wrong in aipy_extracts.__init__. Please "
                "file an issue in our GitHub issue log so that we can help: "
                "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues."
                " Developer info: unknown corrmode."
            )
        # when reading mutliple files we may get a numpy array of file names
        # numpy casts arrays as np.str_ and cython does not like this
        self.filename = str(filename)
        _miriad.UV.__init__(self, self.filename, status, corrmode)

        self.status = status
        self.nchan = _miriad.MAXCHAN

        if status == "old":
            self.vartable = self._gen_vartable()
            self.read()
            self.rewind()  # Update variables for the user
            # Karto: it does not seem possible to end up in a situation where nchan
            # is missing from the vartable -- the basic uvwrite utility checks
            # that value so that knows if nchan has changes (and can update
            # accordingly), so you would have to use something _outside_ of the
            # MIRIAD tools to create such an error.
            with contextlib.suppress(KeyError):
                self.nchan = self["nchan"]
        else:
            self.vartable = {"corr": corrmode}

    def _gen_vartable(self):
        """
        Generate table of variables and types from the vartable header.

        Returns
        -------
        dict
            variables and types from the vartable header
        """
        vartable = {}
        for line in self._rdhd("vartable").split("\n"):
            try:
                var_type, name = line.split()
                vartable[name] = var_type
            except ValueError:
                pass

        return vartable

    def variables(self):
        """
        Get the list of available variables.

        Returns
        -------
        list of str
            list of available variables
        """
        return list(self.vartable.keys())

    def items(self):
        """
        Get the list of available header items.

        Returns
        -------
        list of str
            list of available header items
        """
        items = []

        for i in itemtable:
            try:
                _miriad.hdaccess(self.haccess(i, "read"))
                items.append(i)
            except OSError:
                pass
        return items

    def _rdhd(self, name):
        """
        Provide read access to header items via low-level calls.

        Parameters
        ----------
        name : str
            name of header item

        Returns
        -------
        str or int or float
            value of header item
        """
        itype = itemtable[name]

        if itype == "?":
            return self._rdhd_special(name)

        h = self.haccess(name, "read")
        rv = []

        if len(itype) == 1:
            if itype == "a":
                offset = 0
            else:
                t, offset = _miriad.hread_init(h)
                if itype != t:  # pragma: no cover
                    raise RuntimeError(
                        "Something went wrong in aipy_extracts._rdhd. Please "
                        "file an issue in our GitHub issue log so that we can help: "
                        "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues."
                        " Developer info: itype != t"
                    )

            while True:
                try:
                    c, o = _miriad.hread(h, offset, itype)
                except OSError:
                    break

                if itype == "a":
                    # hread will always return a byte array for itype="a", which we will
                    # want to convert into a string.
                    c = str(c[:o], "utf-8")

                rv.append(c)
                offset += o

            if itype == "a":
                rv = "".join(rv)
        else:
            t, offset = _miriad.hread_init(h)
            if t != "b":  # pragma: no cover
                raise RuntimeError(
                    "Something went wrong in aipy_extracts._rdhd. Please "
                    "file an issue in our GitHub issue log so that we can help: "
                    "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues."
                    " Developer info: t != b."
                )

            for t in itype:
                v, o = _miriad.hread(h, offset, t)
                rv.append(v)
                offset += o

        _miriad.hdaccess(h)

        if len(rv) == 1:
            return rv[0]
        elif isinstance(rv, str):
            return rv
        else:
            return np.array(rv)

    def _wrhd(self, name, val):
        """Provide write access to header items via low-level calls."""
        item_type = itemtable[name]

        if item_type == "?":
            return self._wrhd_special(name, val)

        h = self.haccess(name, "write")

        if len(item_type) == 1:
            try:
                len(val)
            except TypeError:
                val = [val]

            if item_type == "a":
                offset = 0
            else:
                offset = _miriad.hwrite_init(h, item_type)

            for v in val:
                offset += _miriad.hwrite(h, offset, v, item_type)
        else:
            offset = _miriad.hwrite_init(h, "b")
            for v, t in zip(val, item_type, strict=True):
                offset += _miriad.hwrite(h, offset, v, t)

        _miriad.hdaccess(h)

    def _rdhd_special(self, name):
        """Provide read access to special header items of type '?' to _rdhd."""
        if name == "freqs":
            h = self.haccess(name, "read")
            c, o = _miriad.hread(h, 0, "i")
            rv = [c]
            offset = 8

            while True:
                try:
                    c, o = _miriad.hread(h, offset, "i")
                    rv.append(c)
                    offset += 8

                    c, o = _miriad.hread(h, offset, "d")
                    rv.append(c)
                    offset += 8

                    c, o = _miriad.hread(h, offset, "d")
                    rv.append(c)
                    offset += 8
                except OSError:
                    break

            _miriad.hdaccess(h)
            return rv
        elif name == "gains":
            h = self.haccess(name, "read")
            offset = 8
            nsolns = self["nsols"]
            ngains = self["ngains"]
            nants = self["nants"]
            ntau = self["ntau"]
            nfeeds = self["nfeeds"]
            timestamps = np.empty(nsolns, dtype=float)
            soln_arr = np.empty((nsolns, ngains), dtype=np.complex64)

            for idx in range(nsolns):
                timestamps[idx], o = _miriad.hread(h, offset, "d")
                offset += o
                for jdx in range(ngains):
                    soln_arr[idx, jdx], o = _miriad.hread(h, offset, "c")
                    offset += o
            _miriad.hdaccess(h)
            soln_arr = np.reshape(soln_arr, (nsolns, nants, nfeeds + ntau))
            gain_arr = None if (nfeeds == 0) else soln_arr[:, :, :nfeeds]
            delay_arr = None if (ntau == 0) else soln_arr[:, :, nfeeds:].imag

            return timestamps, gain_arr, delay_arr
        elif name == "bandpass":
            h = self.haccess(name, "read")
            offset = 8
            nvals = self["nchan0"] * self["nfeeds"] * self["nants"]
            nsolns = self["nbpsols"]
            timestamps = np.empty(nsolns, dtype=np.float64)
            soln_arr = np.empty((nsolns, nvals), dtype=np.complex64)
            for idx in range(nsolns):
                for jdx in range(nvals):
                    soln_arr[idx, jdx], o = _miriad.hread(h, offset, "c")
                    offset += o
                timestamps[idx], o = _miriad.hread(h, offset, "d")
                offset += o
            _miriad.hdaccess(h)
            soln_arr = np.reshape(
                soln_arr, (nsolns, self["nants"], self["nfeeds"], self["nchan0"])
            )
            return timestamps, soln_arr
        elif name == "leakage":
            h = self.haccess(name, "read")
            offset = 8
            nvals = self["nants"] * self["nfeeds"]
            soln_arr = np.empty(nvals, dtype=np.complex64)
            for idx in range(nvals):
                soln_arr[idx], o = _miriad.hread(h, offset, "c")
                offset += o
            _miriad.hdaccess(h)
            soln_arr = np.reshape(soln_arr, (self["nants"], self["nfeeds"]))
            return soln_arr
        else:
            raise ValueError("Unknown special header: " + name)

    def _hinit_special(self, name):
        """Initialize a special header table."""
        # Initialize an 8-byte entry so that uvio can "latch"
        np.int64(0).tofile(os.path.join(self.filename, name))
        handle = self.haccess(name, "append")
        offset = 8
        return handle, offset

    def _wrhd_special(self, name, val):
        """Provide write access to special header items of type '?' to _wrhd."""
        if name == "freqs":
            h = self.haccess(name, "write")
            _miriad.hwrite(h, 0, val[0], "i")
            offset = 8

            for i, v in enumerate(val[1:]):
                if i % 3 == 0:
                    _miriad.hwrite(h, offset, v, "i")
                else:
                    _miriad.hwrite(h, offset, v, "d")
                offset += 8

            _miriad.hdaccess(h)
        elif name == "leakage":
            # Initialize the leakage table
            handle, offset = self._hinit_special(name)
            for item in val.flat:
                _miriad.hwrite(handle, offset, item, "c")
                offset += 8
            _miriad.hdaccess(handle)
        elif name == "bandpass":
            handle, offset = self._hinit_special(name)
            timestamps, soln_arr = val
            nsolns = soln_arr.shape[0]
            for idx in range(nsolns):
                for item in soln_arr[idx].flat:
                    _miriad.hwrite(handle, offset, item, "c")
                    offset += 8
                _miriad.hwrite(handle, offset, timestamps[idx], "d")
                offset += 8
            self["nbpsols"] = nsolns
            _miriad.hdaccess(handle)
        elif name == "gains":
            handle, offset = self._hinit_special(name)
            timestamps, gain_arr, delay_arr = val
            nfeeds = ntau = val_arr = 0
            if gain_arr is not None:
                nfeeds = gain_arr.shape[2]
                val_arr = gain_arr
            if delay_arr is not None:
                temp_arr = np.zeros_like(delay_arr, dtype=np.complex64)
                temp_arr.imag[:] = delay_arr
                delay_arr = temp_arr
                ntau = delay_arr.shape[2]
                val_arr = delay_arr
            if gain_arr is not None and delay_arr is not None:
                # Miriad stores the tau term alongside the gains for each antenna, so
                # the two have to be interleaved along the feed axis.
                val_arr = np.concatenate((gain_arr, delay_arr), axis=2)

            nsolns = len(timestamps)
            for idx in range(nsolns):
                _miriad.hwrite(handle, offset, timestamps[idx], "d")
                offset += 8
                for item in val_arr[idx].flat:
                    _miriad.hwrite(handle, offset, item, "c")
                    offset += 8

            self["ngains"] = val_arr.size // nsolns
            self["nfeeds"] = nfeeds
            self["ntau"] = ntau
            self["nsols"] = nsolns
            _miriad.hdaccess(handle)
        else:
            raise ValueError("Unknown special header: " + name)

    def __getitem__(self, name):
        """Allow access to variables and header items via ``uv[name]``."""
        try:
            var_type = self.vartable[name]
            return self._rdvr(name, var_type)
        except KeyError:
            var_type = itemtable[name]
            return self._rdhd(name)

    def __setitem__(self, name, val):
        """Allow setting variables and header items via ``uv[name] = val``."""
        try:
            var_type = self.vartable[name]
            self._wrvr(name, var_type, val)
        except KeyError:
            self._wrhd(name, val)

    def select(self, name, n1, n2, include=True):
        """
        Choose which data are returned by read().

        Parameters
        ----------
        name : str
            This can be: 'decimate', 'time', 'antennae', 'visibility',
            'uvrange', 'pointing', 'amplitude', 'window', 'or', 'dra',
            'ddec', 'uvnrange', 'increment', 'ra', 'dec', 'and', 'clear',
            'on', 'polarization', 'shadow', 'auto', 'dazim', 'delev'
        n1, n2 : int
            Generally this is the range of values to select. For
            'antennae', this is the two antennae pair to select
            (indexed from 0); a -1 indicates 'all antennae'.
            For 'decimate', n1 is every Nth integration to use, and
            n2 is which integration within a block of N to use.
            For 'shadow', a zero indicates use 'antdiam' variable.
            For 'on', 'window', 'polarization', 'increment', 'shadow' only
            p1 is used.
            For 'and', 'or', 'clear', 'auto' p1 and p2 are ignored.
        include : bool
            If true, the data is selected. If false, the data is
            discarded. Ignored for 'and', 'or', 'clear'.
        """
        if name == "antennae":
            n1 += 1
            n2 += 1
        self._select(name, float(n1), float(n2), int(include))

    def read(self, raw=False):
        """
        Return the next data record.

        Calling this function causes variables to change to
        reflect the record which this function returns.

        Parameters
        ----------
        raw : bool
            if True data and flags are returned seperately

        Returns
        -------
        preamble : tuple
            (uvw, t, (i,j)), where uvw is an array of u,v,w, t is the
            Julian date, and (i,j) is an antenna pair
        data : ndarray or masked array
            ndarray if raw is True, otherwise a masked array
        flags : ndarray
            only returned if raw is True
        """
        preamble, data, flags, nread = self.raw_read(self.nchan)

        if nread == 0:
            raise OSError("No data read")

        flags = np.logical_not(flags)

        if raw:
            return preamble, data, flags
        return preamble, np.ma.array(data, mask=flags)

    def all_data(self, raw=False):
        """
        Provide an iterator over preamble, data.

        Allows constructs like: ``for preamble, data in uv.all_data(): ...``.

        Parameters
        ----------
        raw : bool
            If True data and flags are returned seperately.

        Returns
        -------
        preamble : tuple
            (uvw, t, (i,j)), where uvw is an array of u,v,w, t is the
            Julian date, and (i,j) is an antenna pair
        data : ndarray or masked array of complex
            ndarray if raw is True, otherwise a masked array
        flags : ndarray
            only returned if raw is True
        """
        while True:
            try:
                yield self.read(raw=raw)
            except OSError:
                break

    def write(self, preamble, data, flags=None):
        """
        Write the next data record.

        Parameters
        ----------
        preamble : tuple
            (uvw, t, (i,j)), where uvw is an array of u,v,w, t is the
            Julian date, and (i,j) is an antenna pair
        data : masked array of complex
            spectra for this record
        """
        if data is None:
            return
        if flags is not None:
            flags = np.logical_not(flags)
        elif len(data.mask.shape) == 0:
            flags = np.ones(data.shape)
            # Setting this to false will instantiate the mask, but keep all values
            # unmasked (as expected)
            data.mask = False
        else:
            flags = np.logical_not(data.mask)
            data = data.data

        self.raw_write(preamble, data.astype(np.complex64), flags.astype(np.int32))

    def init_from_uv(self, uv, override=None, exclude=None):
        """
        Initialize header items and variables from another UV.

        Those in override will be overwritten by override[k], and tracking will
        be turned off (meaning they will not be updated in pipe()). Those in
        exclude are omitted completely.

        Parameters
        ----------
        uv : UV object
            Miriad data set object to initialize from
        override : dict
            variables with values to overwrite
        exclude : list
            list of variable to exclude
        """
        if override is None:
            override = {}

        if exclude is None:
            exclude = []

        for k in uv.items():
            if k in exclude:
                continue
            elif k in override:
                self._wrhd(k, override[k])
            else:
                self._wrhd(k, uv[k])

        self.vartable = {}

        for k in uv.variables():
            if k in exclude:
                continue
            elif k == "corr":
                # I don't understand why reading 'corr' segfaults miriad,
                # but it does.  This is a cludgy work-around.
                continue
            elif k in override:
                self.vartable[k] = uv.vartable[k]
                self._wrvr(k, uv.vartable[k], override[k])
            else:
                self.vartable[k] = uv.vartable[k]
                self._wrvr(k, uv.vartable[k], uv[k])
                uv.trackvr(k, "c")  # Set to copy when copyvr() called

    def pipe(self, uv, mfunc=_uv_pipe_default_action, append2hist="", raw=False):
        """
        Pipe in data from another UV.

        Uses the function ``mfunc(uv, preamble, data)``, which should return
        ``(preamble, data)``. If mfunc is not provided, the dataset will just be
        cloned, and if the returned data is None, it will be omitted.

        Parameters
        ----------
        uv : UV object
            Miriad data set object to pipe from
        mfunc : function
            function that defines how the data are piped.
            ``mfunc(uv, preamble, data)`` should return ``(preamble, data)``.
            Default is ``_uv_pipe_default_action`` which just clones the dataset.
        append2hist : str
            string to append to history
        raw : bool
            if True data and flags are piped seperately
        """
        self._wrhd("history", self["history"] + append2hist)

        if raw:
            for p, d, f in uv.all_data(raw=raw):
                np, nd, nf = mfunc(uv, p, d, f)
                self.copyvr(uv)
                self.write(np, nd, nf)
        else:
            for p, d in uv.all_data():
                np, nd = mfunc(uv, p, d)
                self.copyvr(uv)
                self.write(np, nd)

    def add_var(self, name, var_type):
        """
        Add a variable of the specified type to a UV file.

        Parameters
        ----------
        name : str
            name of header item to add
        var_type : str
            string indicating the variable type (e.g. 'a', 'i', 'd')
        """
        self.vartable[name] = var_type

    def get_freq_axis(self):
        """
        Construct the frequency axis of the underlying visibility data.

        Returns
        -------
        freq_array : ndarray of float
            Channel frequencies in Hz, shape (nchan,).
        channel_width : ndarray of float
            Channel widths in Hz, shape (nchan,).
        flex_spw_id_array : ndarray of int
            Spectral window number for each channel, shape (nchan,).
        spw_array : ndarray of int
            The spectral window numbers, shape (nspect,).
        """
        nspws = self["nspect"]
        if nspws > 1:
            # Channel widths are described per spw, just need to expand it out to be
            # for each frequency channel.
            channel_width = np.concatenate(
                tuple(
                    np.full(nchan, 1e9 * np.abs(chan_width), dtype=np.float64)
                    for chan_width, nchan in zip(
                        self["sdf"], self["nschan"], strict=True
                    )
                )
            )
            freq_array = np.concatenate(
                tuple(
                    (chan_width * np.arange(nchan, dtype=np.float64) + sfreq) * 1e9
                    for chan_width, nchan, sfreq in zip(
                        self["sdf"], self["nschan"], self["sfreq"], strict=True
                    )
                )
            )
            # TODO: Fix this to capture unsorted spectra
            flex_spw_id_array = np.concatenate(
                tuple(
                    np.full(nchan, idx, dtype=int)
                    for idx, nchan in zip(range(nspws), self["nschan"], strict=True)
                )
            )
        else:
            # sdf (delta freq) and sfreq (chan0 freq) are both in GHz
            nchan = self["nchan"]
            flex_spw_id_array = np.zeros(nchan, dtype=int)
            freq_array = 1e9 * (np.arange(nchan) * self["sdf"] + self["sfreq"])
            # Do the units and potential sign conversion for channel_width
            channel_width = np.full(nchan, np.abs(self["sdf"] * 1e9))

        return freq_array, channel_width, flex_spw_id_array, np.arange(nspws)

    def get_data_antennas(self):
        """
        Determine which antennas actually appear in the visibilities.

        Returns
        -------
        list of int
            Sorted antenna numbers that appear in the data.

        """
        ants = set()
        while True:
            try:
                bl_ants = self.read(raw=True)[0][2]
            except OSError:
                # Raised once the records have been exhausted.
                break
            ants.update(bl_ants)
        self.rewind()

        return sorted(int(ant) for ant in ants)

    def get_telescope(
        self, *, telescope=None, sorted_unique_ants=None, correct_lat_lon=True
    ):
        """
        Build a Telescope object from the metadata recorded in a Miriad data set.

        Parameters
        ----------
        telescope : Telescope
            An existing object to populate, which will preserve attributes not directly
            set by this method (e.g., `Telescope.instrument`). Default is None, which
            creates a new Telescope object from scratch.
        sorted_unique_ants : list of int
            The antennas that actually appear in the visibilities, used to work out
            which antennas to keep track of. Note this is needed because MIRIAD uses
            antenna number to index antenna-based information (such that if only
            antennas 0 and 100 are present, values for antennas 1-99 are populated with
            zeros). Default is None, in which case `get_data_antennas` is used to
            determine the antennas.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.

        Returns
        -------
        telescope : Telescope
            The telescope described by the data set.

        """
        from ..telescopes import Telescope

        if sorted_unique_ants is None:
            sorted_unique_ants = self.get_data_antennas()
        if telescope is None:
            telescope = Telescope()
        telescope.name = self["telescop"].replace("\x00", "")
        if "instrume" in self.vartable:
            telescope.instrument = self["instrume"].replace("\x00", "")
        else:
            # set instrument to the telescope name if not set
            telescope.instrument = telescope.name

        self._load_telescope_coords(telescope, correct_lat_lon=correct_lat_lon)
        self._load_antpos(telescope, sorted_unique_ants=sorted_unique_ants)
        self._load_feeds(telescope)

        return telescope

    def _load_feeds(self, telescope):
        """
        Load the mount and feed description onto a Telescope object, if recorded.

        Miriad records `mount` for every one of the `nants` antennas, while pyuvdata
        writes these out per antenna that it is tracking, so anything sized by `nants`
        has to be cut down to the antennas that were kept.

        Parameters
        ----------
        telescope : Telescope
            The object to load the mount and feed information onto. The antennas must
            already have been loaded onto it by `_load_antpos`.

        """
        from .. import utils

        nants = self["nants"]
        ant_nums = telescope.antenna_numbers

        def _select_ants(arr):
            """Cut an array indexed by Miriad antenna number down to the ants used."""
            return arr[ant_nums] if len(arr) == nants else arr

        if "mount" in self.vartable:
            mount = self["mount"]
            if not isinstance(mount, np.ndarray):
                mount = np.full(nants, mount)
            telescope.mount_type = [
                utils.antenna.MOUNT_NUM2STR_DICT[item] for item in _select_ants(mount)
            ]
        if all(item in self.vartable for item in ["nfeeds", "feedarr", "feedang"]):
            telescope.Nfeeds = self["nfeeds"]
            telescope.feed_array = _select_ants(
                np.array(
                    [
                        item.strip()
                        for item in self["feedarr"].replace("\x00", "")[1:-1].split(",")
                    ],
                    dtype=np.object_,
                ).reshape(-1, telescope.Nfeeds)
            )
            telescope.feed_angle = _select_ants(
                self["feedang"].reshape(-1, telescope.Nfeeds)
            )

    def _load_telescope_coords(self, telescope, *, correct_lat_lon=True):
        """
        Load telescope lat, lon, alt coordinates onto a Telescope object.

        Parameters
        ----------
        telescope : Telescope
            The object to load the coordinates onto.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.

        """
        from ..telescopes import known_telescope_location

        latitude = self["latitud"]  # in units of radians
        longitude = self["longitu"]

        # Catch a weird case where where sometimes long is wrapped like RA (0 -> 2pi
        # instead of -pi -> pi)
        if longitude > np.pi:
            longitude -= 2 * np.pi
        try:
            altitude = self["altitude"]
            telescope.location = EarthLocation.from_geodetic(
                lat=latitude * units.rad,
                lon=longitude * units.rad,
                height=altitude * units.m,
            )
        except KeyError:
            # get info from known telescopes.
            # Check to make sure the lat/lon values match reasonably well
            try:
                telescope_loc = known_telescope_location(telescope.name)
            except ValueError:
                telescope_loc = None
            if telescope_loc is not None:
                tol = 2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0)  # 1mas in radians
                lat_close = np.isclose(
                    telescope_loc.lat.rad, latitude, rtol=0, atol=tol
                )
                lon_close = np.isclose(
                    telescope_loc.lon.rad, longitude, rtol=0, atol=tol
                )
                if correct_lat_lon:
                    telescope.location = telescope_loc
                else:
                    telescope.location = EarthLocation.from_geodetic(
                        lat=latitude * units.rad,
                        lon=longitude * units.rad,
                        height=telescope_loc.height,
                    )
                if lat_close and lon_close:
                    if correct_lat_lon:
                        warnings.warn(
                            "Altitude is not present in Miriad file, "
                            f"using known location values for {telescope.name}."
                        )
                    else:
                        warnings.warn(
                            "Altitude is not present in Miriad file, "
                            "using known location altitude value "
                            f"for {telescope.name} and lat/lon from file."
                        )
                else:
                    warn_string = "Altitude is not present in file "
                    if not lat_close and not lon_close:
                        warn_string = (
                            warn_string
                            + "and latitude and longitude values do not match values "
                        )
                    else:
                        if not lat_close:
                            warn_string = (
                                warn_string + "and latitude value does not match value "
                            )
                        else:
                            warn_string = (
                                warn_string
                                + "and longitude value does not match value "
                            )
                    if correct_lat_lon:
                        warn_string = (
                            warn_string + f"for {telescope.name} in known "
                            "telescopes. Using values from known telescopes."
                        )
                        warnings.warn(warn_string)
                    else:
                        warn_string = (
                            warn_string + f"for {telescope.name} in known "
                            "telescopes. Using altitude value from known "
                            "telescopes and lat/lon from file."
                        )
                        warnings.warn(warn_string)
            else:
                warnings.warn(
                    "Altitude is not present in Miriad file, and "
                    f"telescope {telescope.name} is not in "
                    "known_telescopes. Telescope location will be "
                    "set using antenna positions."
                )

    def _load_antpos(self, telescope, *, sorted_unique_ants=None):
        """
        Load antennas and their positions onto a Telescope object.

        Parameters
        ----------
        telescope : Telescope
            The object to load the antennas and positions onto.
        sorted_unique_ants : list of int
            The antennas that actually appear in the visibilities.

        """
        from .. import utils

        latitude = self["latitud"]  # in units of radians
        longitude = self["longitu"]

        # Miriad has no way to keep track of antenna numbers, so the antenna
        # numbers are simply the index for each antenna in any array that
        # describes antenna attributes (e.g. antpos for the antenna_positions).
        # Therefore on write, nants (which gives the size of the antpos array)
        # needs to be increased to be the max value of antenna_numbers+1 and the
        # antpos array needs to be inflated with zeros at locations where we
        # don't have antenna information. These inflations need to be undone at
        # read. If the file was written by pyuvdata, then the variable antnums
        # will be present and we can use it, otherwise we need to test for zeros
        # in the antpos array and/or antennas with no visibilities.
        try:
            # The antnums variable will only exist if the file was written with
            # pyuvdata.
            # For some reason Miriad doesn't handle an array of integers properly,
            # so we convert to floats on write and back here
            telescope.antenna_numbers = self["antnums"].astype(int)
            telescope.Nants = len(telescope.antenna_numbers)
        except KeyError:
            telescope.antenna_numbers = None
            telescope.Nants = None

        nants = self["nants"]
        try:
            # Miriad stores antpos values in units of ns, pyuvdata uses meters.
            antpos = self["antpos"].reshape(3, nants).T * const.c.to_value("m/ns")

            # first figure out what are good antenna positions so we can only
            # use the non-zero ones to evaluate position information
            antpos_length = np.sqrt(np.sum(np.abs(antpos) ** 2, axis=1))
            good_antpos = np.where(antpos_length > 0)[0]
            absolute_positions = False
            if any(good_antpos):
                mean_antpos_length = np.mean(antpos_length[good_antpos])
                if mean_antpos_length > 6.35e6 and mean_antpos_length < 6.39e6:
                    absolute_positions = True

            # Miriad stores antpos values in a rotated ECEF coordinate system
            # where the x-axis goes through the local meridan. Need to convert
            # these positions back to standard ECEF and if they are absolute
            # positions, subtract off the telescope position to make them
            # relative to the array center.
            ecef_antpos = utils.ECEF_from_rotECEF(antpos, longitude)

            if telescope.location is not None:
                if absolute_positions:
                    rel_ecef_antpos = ecef_antpos - telescope._location.xyz()
                    # maintain zeros because they mark missing data
                    rel_ecef_antpos[np.where(antpos_length == 0)[0]] = ecef_antpos[
                        np.where(antpos_length == 0)[0]
                    ]
                else:
                    rel_ecef_antpos = ecef_antpos
            else:
                telescope.location = EarthLocation.from_geocentric(
                    *np.mean(ecef_antpos[good_antpos, :], axis=0) * units.m
                )
                valid_location = utils.coordinates.check_surface_based_positions(
                    telescope_loc=telescope.location,
                    raise_error=False,
                    raise_warning=False,
                )

                # check to see if this could be a valid telescope location
                if valid_location:
                    mean_lon, mean_lat, mean_alt = telescope.location.geodetic
                    mean_lat = mean_lat.rad
                    mean_lon = mean_lon.rad
                    mean_alt = mean_alt.to_value("m")
                    tol = 2 * np.pi / (60.0 * 60.0 * 24.0)  # 1 arcsecond in radians
                    mean_lat_close = np.isclose(mean_lat, latitude, rtol=0, atol=tol)
                    mean_lon_close = np.isclose(mean_lon, longitude, rtol=0, atol=tol)

                    if mean_lat_close and mean_lon_close:
                        # this looks like a valid telescope location, and the
                        # mean antenna lat & lon values are close. Set the
                        # telescope location using the file lat/lons and the mean alt.
                        # Then subtract it off of the antenna positions
                        warnings.warn(
                            "Telescope location is not set, but antenna "
                            "positions are present. Mean antenna latitude and "
                            "longitude values match file values, so "
                            "telescope_position will be set using the "
                            "mean of the antenna altitudes"
                        )
                        telescope.location = EarthLocation.from_geodetic(
                            lat=latitude * units.rad,
                            lon=longitude * units.rad,
                            height=mean_alt * units.m,
                        )
                        rel_ecef_antpos = ecef_antpos - telescope._location.xyz()

                    else:
                        # this looks like a valid telescope location, but the
                        # mean antenna lat & lon values are not close. Set the
                        # telescope location using the file lat/lons at sea level.
                        # Then subtract it off of the antenna positions
                        telescope.location = EarthLocation.from_geodetic(
                            lat=latitude * units.rad, lon=longitude * units.rad
                        )
                        warn_string = (
                            "Telescope location is set at sealevel at "
                            "the file lat/lon coordinates. Antenna "
                            "positions are present, but the mean "
                            "antenna "
                        )
                        rel_ecef_antpos = ecef_antpos - telescope._location.xyz()

                        if not mean_lat_close and not mean_lon_close:
                            warn_string += (
                                "latitude and longitude values do not "
                                "match file values so they are not used "
                                "for altitude."
                            )
                        elif not mean_lat_close:
                            warn_string += (
                                "latitude value does not "
                                "match file values so they are not used "
                                "for altitude."
                            )
                        else:
                            warn_string += (
                                "longitude value does not "
                                "match file values so they are not used "
                                "for altitude."
                            )
                        warnings.warn(warn_string)

                else:
                    # This does not give a valid telescope location. Instead
                    # calculate it from the file lat/lon and sea level for altitude
                    telescope.location = EarthLocation.from_geodetic(
                        lat=latitude * units.rad, lon=longitude * units.rad
                    )
                    warn_string = (
                        "Telescope location is set at sealevel at the file lat/lon "
                        "coordinates. Antenna positions are present, but the mean "
                        "antenna position does not give a telescope location on the "
                        "surface of the earth."
                    )
                    if absolute_positions:
                        rel_ecef_antpos = ecef_antpos - telescope._location.xyz()
                    else:
                        warn_string += (
                            " Antenna positions do not appear to be "
                            "on the surface of the earth and will be treated "
                            "as relative."
                        )
                        rel_ecef_antpos = ecef_antpos

                    warnings.warn(warn_string)

            if telescope.Nants is not None:
                # in this case there is an antnums variable
                # (meaning that the file was written with pyuvdata), so we'll use it
                if nants == telescope.Nants:
                    # no inflation, so just use the positions
                    telescope.antenna_positions = rel_ecef_antpos
                else:
                    # there is some inflation, just use the ones that appear in antnums
                    telescope.antenna_positions = np.zeros(
                        (telescope.Nants, 3), dtype=antpos.dtype
                    )
                    for ai, num in enumerate(telescope.antenna_numbers):
                        telescope.antenna_positions[ai, :] = rel_ecef_antpos[num, :]
            else:
                # there is no antnums variable (meaning that this file was not
                # written by pyuvdata), so we test for antennas with non-zero
                # positions and/or that appear in the visibility data
                # (meaning that they have entries in ant_1_array or ant_2_array)
                antpos_length = np.sqrt(np.sum(np.abs(antpos) ** 2, axis=1))
                good_antpos = np.where(antpos_length > 0)[0]
                # take the union of the antennas with good positions (good_antpos)
                # and the antennas that have visisbilities (sorted_unique_ants)
                # if there are antennas with visibilities but zeroed positions
                # we issue a warning below
                if sorted_unique_ants is not None:
                    ants_use = set(good_antpos).union(sorted_unique_ants)
                else:
                    ants_use = set(good_antpos)
                # ants_use are the antennas we'll keep track of in the UVData
                # object, so they dictate Nants_telescope
                telescope.Nants = len(ants_use)
                telescope.antenna_numbers = np.array(list(ants_use))
                telescope.antenna_positions = np.zeros(
                    (telescope.Nants, 3), dtype=rel_ecef_antpos.dtype
                )
                for ai, num in enumerate(telescope.antenna_numbers):
                    if antpos_length[num] == 0:
                        warnings.warn(
                            f"antenna number {num} has visibilities "
                            "associated with it, but it has a position"
                            " of (0,0,0)"
                        )
                    else:
                        # leave bad locations as zeros to make them obvious
                        telescope.antenna_positions[ai, :] = rel_ecef_antpos[num, :]

        except KeyError:
            # there is no antpos variable
            warnings.warn("Antenna positions are not present in the file.")
            telescope.antenna_positions = None

            if telescope.location is None:
                telescope.location = EarthLocation.from_geodetic(
                    lat=latitude * units.rad, lon=longitude * units.rad
                )
                warnings.warn(
                    "Telescope location is set at sealevel at the file lat/lon "
                    "coordinates because neither altitude nor antenna positions "
                    "are present in the file."
                )

        if telescope.antenna_numbers is None and sorted_unique_ants is not None:
            # there are no antenna_numbers or antenna_positions, so just use
            # the antennas present in the visibilities
            # (Nants_data will therefore match Nants_telescope)
            telescope.antenna_numbers = np.array(sorted_unique_ants)
            telescope.Nants = len(telescope.antenna_numbers)

        # antenna names is a foreign concept in miriad but required in other formats.
        try:
            # Here we deal with the way pyuvdata tacks it on to keep the
            # name information if we have it:
            # make it into one long comma-separated string
            ant_name_var = self["antnames"]
            ant_name_str = ant_name_var.replace("\x00", "")
            ant_name_list = ant_name_str[1:-1].split(", ")
            telescope.antenna_names = ant_name_list
        except KeyError:
            if telescope.antenna_numbers is not None:
                telescope.antenna_names = telescope.antenna_numbers.astype(str).tolist()

        # check for antenna diameters
        try:
            telescope.antenna_diameters = self["antdiam"]
        except KeyError:
            # backwards compatibility for when keyword was 'diameter'
            with contextlib.suppress(KeyError):
                telescope.antenna_diameters = self["diameter"]
        if telescope.antenna_diameters is not None:
            telescope.antenna_diameters = telescope.antenna_diameters * np.ones(
                telescope.Nants, dtype=np.float64
            )
