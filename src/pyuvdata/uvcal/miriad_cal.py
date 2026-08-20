# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading Miriad calibration tables."""

import os
import warnings

import numpy as np
from docstring_parser import DocstringStyle

from ..docstrings import copy_replace_short_description
from . import UVCal

__all__ = ["MiriadCal"]


class MiriadCal(UVCal):
    """
    Defines a Miriad-specific subclass of UVCal for reading Miriad calibration tables.

    This class should not be interacted with directly, instead use the
    read_miriad_cal method on the UVCal class.

    """

    @copy_replace_short_description(
        UVCal.read_miriad_cal, style=DocstringStyle.NUMPYDOC
    )
    def read_miriad_cal(
        self,
        filepath,
        *,
        soln_type=None,
        default_mount_type="other",
        default_x_orientation=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        astrometry_library=None,
    ):
        """Read in calibration solutions from a Miriad data set."""
        from ..uvdata import aipy_extracts

        if not os.path.exists(filepath):
            raise OSError(filepath + " not found")

        uv = aipy_extracts.UV(filepath)
        soln_type, read_soln = self._select_soln(uv, soln_type)

        self.history = uv["history"]
        self.filename = [os.path.basename(filepath.rstrip("/"))]
        self._filename.form = (1,)

        self._set_sky()
        self.cal_style = "sky"
        self.sky_catalog = "unknown"
        # Karto: manually confirmed in testing the ATCA test file
        self.gain_convention = "multiply"

        pol_check = uv["pol"]
        nfeeds = uv["nfeeds"]
        if nfeeds == 1:
            # Assume that the Jones type matches the pol listed.
            self.jones_array = np.atleast_1d(pol_check)
        elif nfeeds == 2:
            if pol_check in [-1, -2, -3, -4]:
                # If circular, use circular codes (cross-hand if leakages)
                self.jones_array = np.array(
                    [-3, -4] if soln_type == "leakage" else [-1, -2]
                )
            elif pol_check in [-5, -6, -7, -8]:
                # If linear, use linear codes (cross-hand if leakages)
                self.jones_array = np.array(
                    [-7, -8] if soln_type == "leakage" else [-5, -6]
                )
            else:
                raise ValueError(
                    f"Miriad data set has an unrecognized pol code ({pol_check}), "
                    "cannot determine feed basis of the calibration table."
                )
        else:
            raise ValueError(
                f"MIRIAD data set shows {nfeeds} feeds, expected only 1 or 2."
            )

        self.Njones = self.jones_array.size

        self.history += (
            f"  Read the {soln_type} table from a Miriad data set using pyuvdata."
        )

        # Njones has to be known before the table is read, and the antennas that the
        # table covers have to be known before the telescope is built.
        read_soln(uv)

        # Grab the telescope information from the file.
        uv.get_telescope(telescope=self.telescope, sorted_unique_ants=self.ant_array)

        if self.cal_type == "delay":
            # The reference antenna is the one left with no delay at all.
            ref_check = np.abs(self.delay_array)
        else:
            ref_check = np.abs(np.angle(self.gain_array))

        good_mask = ~self.flag_array

        # Reference antenna is nominally pinned to exactly zero phase, but the solver
        # can leave residuals of a few eps. Use a threshold of 1e-6 times the largest
        # value to determine which antennas might be the refant.
        ref_ant = np.nonzero(
            np.any((ref_check < 1e-6 * np.max(ref_check)) & good_mask, axis=(1, 2, 3))
        )[0]

        if ref_ant.size != 1:
            self.ref_antenna_name = "unknown"
        else:
            ant_idx = np.nonzero(
                self.telescope.antenna_numbers == self.ant_array[ref_ant[0]]
            )[0][0]
            self.ref_antenna_name = self.telescope.antenna_names[ant_idx]

        # Miriad only records the feed orientation if the data set was written by
        # pyuvdata, so default to "east" if nothing is recorded or specified.
        if "xorient" in uv.vartable:
            x_orientation = uv["xorient"].replace("\x00", "")
        elif default_x_orientation is not None:
            x_orientation = default_x_orientation
        else:
            x_orientation = "east"
            warnings.warn('Unknown x_orientation basis for solutions, assuming "east".')

        self.set_lsts_from_time_array(astrometry_library=astrometry_library)
        # Skip the check here since it is run (or deliberately skipped) below.
        self.set_telescope_params(
            x_orientation=x_orientation, mount_type=default_mount_type, run_check=False
        )

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def _select_soln(self, uv, soln_type):
        """
        Validate the requested calibration table against what the data set holds.

        Parameters
        ----------
        uv : aipy_extracts.UV
            An open handle to the Miriad data set.
        soln_type : str or None
            The requested table, or None to select one automatically.

        Returns
        -------
        soln_type : str
            The table to read.
        read_soln : callable
            The method that reads that table.
        """
        # Header item that has to be present, plus the reader, for each table.
        _SOLN_TYPES = {
            "gains": ("nsols", self._read_gains_table),
            "delays": ("ntau", self._read_delays_table),
            "bandpass": ("nchan0", self._read_bandpass_table),
            "leakage": ("leakage", self._read_leakage_table),
        }

        if (soln_type is not None) and (soln_type not in _SOLN_TYPES):
            raise ValueError(
                f"soln_type must be one of {list(_SOLN_TYPES)}, got {soln_type}."
            )

        found = []
        for item, (probe, _) in _SOLN_TYPES.items():
            try:
                # ntau is recorded as zero when no delays were solved for.
                if item == "delays" and uv[probe] < 1:
                    continue
                uv[probe]
            except (KeyError, OSError):
                # Miriad raises OSError rather than KeyError for a missing header item.
                continue

            found.append(item)

        listing = found if found else "no calibration tables"

        if soln_type is None:
            if len(found) != 1:
                raise ValueError(
                    "Cannot determine which calibration table to read: this data set "
                    f"contains {listing}. Set soln_type to select one."
                )
            soln_type = found[0]

        if soln_type not in found:
            raise ValueError(
                f"This Miriad data set has no {soln_type} table, it contains {listing}."
            )

        return soln_type, _SOLN_TYPES[soln_type][1]

    def _set_freq_range_from_vis(self, uv):
        """
        Build a wide-band frequency axis from the spectral layout of the visibilities.

        Parameters
        ----------
        uv : aipy_extracts.UV
            An open handle to the Miriad data set.

        """
        freq_array, channel_width, flex_spw_id_array, spw_array = uv.get_freq_axis()

        self._set_wide_band()
        self.spw_array = spw_array
        self.Nspws = spw_array.size
        self.Nfreqs = 1
        self.freq_range = np.empty((self.Nspws, 2), dtype=float)
        for idx, spw in enumerate(self.spw_array):
            spw_mask = flex_spw_id_array == spw
            spw_freqs = freq_array[spw_mask]
            half_width = 0.5 * channel_width[spw_mask]
            self.freq_range[idx] = [
                np.min(spw_freqs - half_width),
                np.max(spw_freqs + half_width),
            ]

    def _set_times_from_table(self, uv, time_array):
        """
        Record the solution times, and the interval that Miriad declares for them.

        Parameters
        ----------
        uv : aipy_extracts.UV
            An open handle to the Miriad data set.
        time_array : ndarray of float
            Solution timestamps in JD.

        """
        self.time_array = time_array
        self.Ntimes = time_array.size

        interval = uv["interval"] * 86400.0
        # Note that mfcal hardwires interval to half a day rather than recording what
        # was actually used (as evidenced by the ATCA file and evaluating the code), but
        # it always produces a single entry.
        if self.Ntimes > 1:
            spacing = np.median(np.diff(np.sort(time_array))) * 86400.0
            if interval > 2 * spacing:
                # Usually the interval is only long like this b/c it's been mucked with
                # via gpedit -- let the user know since it'll make the int times
                # somewhat meaningless.
                warnings.warn(
                    f"Miriad records a solution interval of {interval:.0f} s, which is "
                    f"much longer than the {spacing:.0f} s spacing between solutions."
                )
        self.integration_time = np.full(self.Ntimes, interval)

    def _read_bandpass_table(self, uv):
        """
        Read the Miriad bandpass table into a frequency resolved gain object.

        Parameters
        ----------
        uv : aipy_extracts.UV
            An open handle to the Miriad data set.

        """
        self._set_gain()

        # Grab the frequency information
        freq_info = np.asarray(uv["freqs"][1:], dtype=float).reshape(-1, 3)
        if freq_info.shape[0] != uv["nspect0"]:  # pragma: no cover
            raise ValueError(
                f"Miriad freqs item describes {freq_info.shape[0]} windows, but "
                f"nspect0 records {uv['nspect0']}."
            )

        nschan = freq_info[:, 0].astype(int)
        if nschan.sum() != uv["nchan0"]:  # pragma: no cover
            raise ValueError(
                f"Miriad freqs item describes {nschan.sum()} channels, but nchan0 "
                f" records {uv['nchan0']}."
            )

        # sfreq and sdf are recorded in GHz, and sdf may be negative.
        sfreq, sdf = freq_info[:, 1], freq_info[:, 2]
        windows = zip(nschan, sfreq, sdf, strict=True)
        self.freq_array = np.concatenate(
            [1e9 * (start + np.arange(nch) * width) for nch, start, width in windows]
        )
        self.channel_width = np.concatenate(
            [
                np.full(nchan, 1e9 * np.abs(width))
                for nchan, width in zip(nschan, sdf, strict=True)
            ]
        )
        self.flex_spw_id_array = np.concatenate(
            [np.full(nchan, idx) for idx, nchan in enumerate(nschan)]
        )

        self.Nfreqs = self.freq_array.size
        self.spw_array = np.unique(self.flex_spw_id_array)
        self.Nspws = self.spw_array.size

        time_array, bandpass = uv["bandpass"]
        soln_mask = np.any(bandpass, axis=(0, 2, 3))
        self._set_times_from_table(uv, time_array)

        # Shape is (Ntimes, Nants, Njones, Nfreqs), and needs to end up as
        # (Nants_data, Nfreqs, Ntimes, Njones).
        self.gain_array = np.transpose(bandpass[:, soln_mask], (1, 3, 0, 2))
        self.flag_array = self.gain_array == 0
        self.ant_array = np.nonzero(soln_mask)[0]
        self.Nants_data = self.ant_array.size

    def _read_gains_table(self, uv, do_delays=False):
        """
        Read the Miriad gains table into a wide-band gain object.

        Parameters
        ----------
        uv : aipy_extracts.UV
            An open handle to the Miriad data set.

        """
        time_array, gain_arr, delay_arr = uv["gains"]

        self._set_times_from_table(uv, time_array)
        # Set the freqs from the vis for gains/delays tables
        self._set_freq_range_from_vis(uv)
        soln_mask = np.any(gain_arr, axis=(0, 2))

        if do_delays:
            self._set_delay()
            # mfcal appears to be the only one that writes the tau term, with tau
            # written in nanoseconds (as seems to be conventional in MIRIAD). Note that
            # delays are applied relative to some given reference frequency (freq0)
            # rather than as an absolute value (w/ a corresponding phase offset).
            delay_arr = -1e-9 * delay_arr / (2 * np.pi)

            # MIRIAD records one delay per antenna regardless of the number of feeds, so
            # so the value applies to every Jones component. Similar to the bandpass, it
            # also appears to only allow for one delay soln to be derived for a given
            # data set (via mfcal, which seems to be the only one to derive these).
            # Shape is (Ntimes, Nants_data, Njones), and needs to end up as
            # (Nants_data, Nspws, Ntimes, Njones).
            delay_arr = np.transpose(delay_arr[:, soln_mask, :], (1, 0, 2))[:, None]
            self.delay_array = np.repeat(
                np.repeat(delay_arr, self.Nspws, axis=1), self.Njones, axis=3
            )

            # Record the reference frequency here as an extra keyword
            self.extra_keywords["FREQ0"] = uv["freq0"]
            self.history += f"  MIRIAD delay freq0 = {uv['freq0']:.6g} GHz."
        else:
            self._set_gain()

            # Shape is (Ntimes, Nants_data, Njones), and needs to end up as
            # (Nants_data, Nspws, Ntimes, Njones).
            self.gain_array = np.repeat(
                np.transpose(gain_arr[:, soln_mask, :], (1, 0, 2))[:, None],
                self.Nspws,
                axis=1,
            )

        # Miriad marks a gain solution bad by storing an exact zero.
        flag_arr = gain_arr[:, soln_mask, :] == 0.0
        self.flag_array = np.repeat(
            np.transpose(flag_arr, (1, 0, 2))[:, None], self.Nspws, axis=1
        )
        if self.cal_type == "delay":
            # Miriad records one delay per antenna, so the flags apply to every Jones.
            self.flag_array = np.repeat(self.flag_array[..., :1], self.Njones, axis=3)
        self.ant_array = np.nonzero(soln_mask)[0]
        self.Nants_data = self.ant_array.size

    def _read_delays_table(self, uv):
        """
        Read the delay terms of the Miriad gains table into a delay object.

        Parameters
        ----------
        uv : aipy_extracts.UV
            An open handle to the Miriad data set.

        """
        self._read_gains_table(uv, do_delays=True)

    def _read_leakage_table(self, uv):
        """
        Read the Miriad leakage table into a cross-handed gain object.

        Parameters
        ----------
        uv : aipy_extracts.UV
            An open handle to the Miriad data set.

        """
        if uv["nfeeds"] != 2:  # pragma: no cover
            raise ValueError("Data set must have `nfeeds=2` for leakage tables")

        self._set_gain()
        self._set_freq_range_from_vis(uv)
        leakage = uv["leakage"]

        # Miriad records a single set of d-terms for the whole data set, somewhat
        # similar to how they're stored in MeasuremenSet format.
        times = None
        for name in ("gains", "bandpass"):
            try:
                times = uv[name][0]
                break
            except (KeyError, OSError):
                continue
        if times is None:
            # Nothing else to go on, use the first record of the data.
            times = np.array([uv["time"]])

        self.Ntimes = 1
        self.time_range = np.array([[times.min(), times.max()]])
        self.integration_time = np.array([(times.max() - times.min()) * 86400.0])

        soln_mask = ~np.all(leakage == 0.0, axis=1)

        # Shape is (Nants_data, Njones), and needs to end up as
        # (Nants_data, Nspws, Ntimes, Njones).
        self.gain_array = np.repeat(leakage[soln_mask, None, None], self.Nspws, axis=1)
        self.flag_array = self.gain_array == 0.0
        self.ant_array = np.nonzero(soln_mask)[0]
        self.Nants_data = self.ant_array.size
