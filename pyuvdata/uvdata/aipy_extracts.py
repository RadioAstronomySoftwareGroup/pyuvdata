# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the 2-clause BSD License

"""Module for low-level interface to MIRIAD files.

This module extracts some Python code from AIPY used in our MIRIAD I/O
routines. It was copied from AIPY commit
6cb5a70876f33dccdd68d4063b076f8d42d9edae, then reformatted. The only items
used by pyuvdata are ``uv_selector`` and ``UV``.

"""
__all__ = ["uv_selector", "UV"]

import numpy as np
import re

try:
    from pyuvdata import _miriad
except ImportError as e:  # pragma: no cover
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
bl_re = "(^(%s_%s|%s),?)" % (ant_re, ant_re, ant_re)


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
            elif ant_str[cnt:].startswith("auto"):
                rv.append(("auto", 1, -1))
            elif ant_str[cnt:].startswith("cross"):
                rv.append(("auto", 0, -1))
            else:
                raise ValueError('Unparsable ant argument "%s"' % ant_str)

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
                if type(i) == str and i.startswith("-"):
                    i = i[1:]  # nibble the - off the string
                    include_i = 0
                else:
                    include_i = 1

                for j in ajs:
                    include = None

                    if type(j) == str and j.startswith("-"):
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
        if type(ants) == str:
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
    "leakage": "c",
    "freq0": "d",
    "freqs": "?",
    "bandpass": "c",
    "nspect0": "i",
    "nchan0": "i",
    "stopt": "d",
    "duration": "d",
}


def _uv_pipe_default_action(uv, p, d):
    return p, d


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
        assert status in ["old", "new", "append"]
        assert corrmode in ["r", "j"]
        # when reading mutliple files we may get a numpy array of file names
        # numpy casts arrays as np.str_ and cython does not like this
        _miriad.UV.__init__(self, str(filename), status, corrmode)

        self.status = status
        self.nchan = _miriad.MAXCHAN

        if status == "old":
            self.vartable = self._gen_vartable()
            self.read()
            self.rewind()  # Update variables for the user
            try:
                self.nchan = self["nchan"]
            except KeyError:
                pass
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
            except IOError:
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
                assert itype == t

            while True:
                try:
                    c, o = _miriad.hread(h, offset, itype)
                except IOError:
                    break

                if itype == "a":
                    try:
                        c = str(c[:o], "utf-8")
                    except TypeError:
                        c = c[:o]

                rv.append(c)
                offset += o

            if itype == "a":
                rv = "".join(rv)
        else:
            t, offset = _miriad.hread_init(h)
            assert t == "b"

            for t in itype:
                v, o = _miriad.hread(h, offset, t)
                rv.append(v)
                offset += o

        _miriad.hdaccess(h)

        if len(rv) == 1:
            return rv[0]
        elif type(rv) == str:
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
            for v, t in zip(val, item_type):
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
                except IOError:
                    break

            _miriad.hdaccess(h)
            return rv
        else:
            raise ValueError("Unknown special header: " + name)

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
            raise IOError("No data read")

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
            except IOError:
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
            data = data.unmask()
        else:
            flags = np.logical_not(data.mask)
            data = data.data

        self.raw_write(preamble, data.astype(np.complex64), flags.astype(np.int32))

    def init_from_uv(self, uv, override={}, exclude=[]):
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
