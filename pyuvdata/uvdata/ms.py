# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Class for reading and writing casa measurement sets.

Requires casacore.
"""
import numpy as np
import os
import warnings
from astropy.time import Time

from .uvdata import UVData
from .. import utils as uvutils

__all__ = ["MS"]

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
try:
    import casacore.tables as tables
    import casacore.tables.tableutil as tableutil
except ImportError as error:  # pragma: no cover
    casa_present = False
    casa_error = error

"""
This dictionary defines the mapping between CASA polarization numbers and
AIPS polarization numbers
"""
# convert from casa polarization integers to pyuvdata
POL_CASA2AIPS_DICT = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: -1,
    6: -3,
    7: -4,
    8: -2,
    9: -5,
    10: -7,
    11: -8,
    12: -6,
}

POL_AIPS2CASA_DICT = {
    aipspol: casapol for casapol, aipspol in POL_CASA2AIPS_DICT.items()
}

VEL_DICT = {
    "REST": 0,
    "LSRK": 1,
    "LSRD": 2,
    "BARY": 3,
    "GEO": 4,
    "TOPO": 5,
    "GALACTO": 6,
    "LGROUP": 7,
    "CMB": 8,
    "Undefined": 64,
}


# In CASA 'J2000' refers to a specific frame -- FK5 w/ an epoch of
# J2000. We'll plug that in here directly, noting that CASA has an
# explicit list of supported reference frames, located here:
# casa.nrao.edu/casadocs/casa-5.0.0/reference-material/coordinate-frames

COORD_UVDATA2CASA_DICT = {
    "J2000": ("fk5", 2000.0),  # mean equator and equinox at J2000.0 (FK5)
    "JNAT": None,  # geocentric natural frame
    "JMEAN": None,  # mean equator and equinox at frame epoch
    "JTRUE": None,  # true equator and equinox at frame epoch
    "APP": ("gcrs", 2000.0),  # apparent geocentric position
    "B1950": ("fk4", 1950.0),  # mean epoch and ecliptic at B1950.0.
    "B1950_VLA": ("fk4", 1979.0),  # mean epoch (1979.9) and ecliptic at B1950.0
    "BMEAN": None,  # mean equator and equinox at frame epoch
    "BTRUE": None,  # true equator and equinox at frame epoch
    "GALACTIC": None,  # Galactic coordinates
    "HADEC": None,  # topocentric HA and declination
    "AZEL": None,  # topocentric Azimuth and Elevation (N through E)
    "AZELSW": None,  # topocentric Azimuth and Elevation (S through W)
    "AZELNE": None,  # topocentric Azimuth and Elevation (N through E)
    "AZELGEO": None,  # geodetic Azimuth and Elevation (N through E)
    "AZELSWGEO": None,  # geodetic Azimuth and Elevation (S through W)
    "AZELNEGEO": None,  # geodetic Azimuth and Elevation (N through E)
    "ECLIPTIC": None,  # ecliptic for J2000 equator and equinox
    "MECLIPTIC": None,  # ecliptic for mean equator of date
    "TECLIPTIC": None,  # ecliptic for true equator of date
    "SUPERGAL": None,  # supergalactic coordinates
    "ITRF": None,  # coordinates wrt ITRF Earth frame
    "TOPO": None,  # apparent topocentric position
    "ICRS": ("icrs", 2000.0),  # International Celestial reference system
}


class MS(UVData):
    """
    Defines a class for reading and writing casa measurement sets.

    This class should not be interacted with directly, instead use the read_ms
    method on the UVData class.

    Attributes
    ----------
    ms_required_extra : list of str
        Names of optional UVParameters that are required for MS
    """

    ms_required_extra = ["datacolumn", "antenna_positions"]

    def _init_ms_file(self, filepath):
        """
        Create a skeleton MS dataset to fill.

        Parameters
        ----------
        filepath : str
            Path to MS to be created.
        """
        # The required_ms_desc returns the defaults for a CASA MS table
        ms_desc = tables.required_ms_desc("MAIN")

        # The tables have a different choice of dataManagerType and dataManagerGroup
        # based on a test ALMA dataset and comparison with what gets generated with
        # a dataset that comes through importuvfits.
        ms_desc["FLAG"].update(
            dataManagerType="TiledShapeStMan", dataManagerGroup="TiledFlag",
        )

        ms_desc["UVW"].update(
            dataManagerType="TiledColumnStMan", dataManagerGroup="TiledUVW",
        )
        # TODO: Can stuff UVFLAG objects into this
        ms_desc["FLAG_CATEGORY"].update(
            dataManagerType="TiledShapeStMan",
            dataManagerGroup="TiledFlagCategory",
            keywords={"CATEGORY": np.array("baddata")},
        )
        ms_desc["WEIGHT"].update(
            dataManagerType="TiledShapeStMan", dataManagerGroup="TiledWgt",
        )
        ms_desc["SIGMA"].update(
            dataManagerType="TiledShapeStMan", dataManagerGroup="TiledSigma",
        )

        # The ALMA default for the next set of columns from the MAIN table use the
        # name of the column as the dataManagerGroup, so we update the casacore
        # defaults accordingly.
        for key in ["ANTENNA1", "ANTENNA2", "DATA_DESC_ID", "FLAG_ROW"]:
            ms_desc[key].update(dataManagerGroup=key)

        # The ALMA default for he next set of columns from the MAIN table use the
        # IncrementalStMan dataMangerType, and so we update the casacore defaults
        # (along with the name dataManagerGroup name to the column name, which is
        # also the apparent default for ALMA).
        incremental_list = [
            "ARRAY_ID",
            "EXPOSURE",
            "FEED1",
            "FEED2",
            "FIELD_ID",
            "INTERVAL",
            "OBSERVATION_ID",
            "PROCESSOR_ID",
            "SCAN_NUMBER",
            "STATE_ID",
            "TIME",
            "TIME_CENTROID",
        ]
        for key in incremental_list:
            ms_desc[key].update(
                dataManagerType="IncrementalStMan", dataManagerGroup=key,
            )

        # TODO: Verify that the casacore defaults for coldesc are satisfactory for
        # the tables and columns below (note that these were columns with apparent
        # discrepancies between a test ALMA dataset and what casacore generated).
        # FEED:FOCUS_LENGTH
        # FIELD
        # POINTING:TARGET
        # POINTING:POINTING_OFFSET
        # POINTING:ENCODER
        # POINTING:ON_SOURCE
        # POINTING:OVER_THE_TOP
        # SPECTRAL_WINDOW:BBC_NO
        # SPECTRAL_WINDOW:ASSOC_SPW_ID
        # SPECTRAL_WINDOW:ASSOC_NATURE
        # SPECTRAL_WINDOW:SDM_WINDOW_FUNCTION
        # SPECTRAL_WINDOW:SDM_NUM_BIN

        # Create a column for the data, which is amusingly enough not actually
        # creaed by default.
        datacoldesc = tables.makearrcoldesc(
            "DATA",
            0.0 + 0.0j,
            valuetype="complex",
            ndim=2,
            datamanagertype="TiledShapeStMan",
            datamanagergroup="TiledData",
            comment="The data column",
        )
        del datacoldesc["desc"]["shape"]
        ms_desc.update(tables.maketabdesc(datacoldesc))

        # Now create a column for the weight spectrum, which we plug nsample_array into
        weightspeccoldesc = tables.makearrcoldesc(
            "WEIGHT_SPECTRUM",
            0.0,
            valuetype="float",
            ndim=2,
            datamanagertype="TiledShapeStMan",
            datamanagergroup="TiledWgtSpectrum",
            comment="Weight for each data point",
        )
        del weightspeccoldesc["desc"]["shape"]

        ms_desc.update(tables.maketabdesc(weightspeccoldesc))

        # Finally, create the dataset, and return a handle to the freshly created
        # skeleton measurement set.
        return tables.default_ms(filepath, ms_desc, tables.makedminfo(ms_desc))

    def _write_ms_antenna(self, filepath):
        """
        Write out the antenna information into a CASA table.

        Parameters
        ----------
        filepath : str
            Path to MS (without ANTENNA suffix).
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        antenna_table = tables.table(filepath + "::ANTENNA", ack=False, readonly=False)

        # Note: measurement sets use the antenna number as an index into the antenna
        # table. This means that if the antenna numbers do not start from 0 and/or are
        # not contiguous, empty rows need to be inserted into the antenna table
        # (this is somewhat similar to miriad)
        nants_table = np.max(self.antenna_numbers) + 1
        antenna_table.addrows(nants_table)

        ant_names_table = [""] * nants_table
        for ai, num in enumerate(self.antenna_numbers):
            ant_names_table[num] = self.antenna_names[ai]

        # There seem to be some variation on whether the antenna names are stored
        # in the NAME or STATION column (importuvfits puts them in the STATION column
        # while Cotter and the MS definition doc puts them in the NAME column).
        # The MS definition doc suggests that antenna names belong in the NAME column
        # and the telescope name belongs in the STATION column (it gives the example of
        # GREENBANK for this column.) so we follow that here. For a reconfigurable
        # array, the STATION can be though of as the "pad" name, which is distinct from
        # the antenna name/number, and nominally fixed in position.
        antenna_table.putcol("NAME", ant_names_table)
        antenna_table.putcol("STATION", ant_names_table)

        # Antenna positions in measurement sets appear to be in absolute ECEF
        ant_pos_absolute = self.antenna_positions + self.telescope_location.reshape(
            1, 3
        )
        ant_pos_table = np.zeros((nants_table, 3), dtype=np.float64)
        for ai, num in enumerate(self.antenna_numbers):
            ant_pos_table[num, :] = ant_pos_absolute[ai, :]

        antenna_table.putcol("POSITION", ant_pos_table)
        if self.antenna_diameters is not None:
            ant_diam_table = np.zeros((nants_table), dtype=np.float64)
            # This is here is suppress an error that arises when one has antennas of
            # different diameters (which CASA can't handle), since otherwise the
            # "padded" antennas have zero diameter (as opposed to any real telescope).
            if len(np.unique(self.antenna_diameters)) == 1:
                ant_diam_table[:] = self.antenna_diameters[0]
            else:
                for ai, num in enumerate(self.antenna_numbers):
                    ant_diam_table[num] = self.antenna_diameters[ai]
            antenna_table.putcol("DISH_DIAMETER", ant_diam_table)
        antenna_table.done()

    def _write_ms_data_description(self, filepath):
        """
        Write out the data description information into a CASA table.

        Parameters
        ----------
        filepath : str
            Path to MS (without DATA_DESCRIPTION suffix).
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        data_descrip_table = tables.table(
            filepath + "::DATA_DESCRIPTION", ack=False, readonly=False
        )

        data_descrip_table.addrows(self.Nspws)

        data_descrip_table.putcol("SPECTRAL_WINDOW_ID", np.arange(self.Nspws))

        data_descrip_table.done()

    def _write_ms_feed(self, filepath):
        """
        Write out the feed information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without FEED suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        feed_table = tables.table(filepath + "::FEED", ack=False, readonly=False)

        nfeeds_table = np.max(self.antenna_numbers) + 1
        feed_table.addrows(nfeeds_table)

        antenna_id_table = np.arange(nfeeds_table, dtype=np.int32)
        feed_table.putcol("ANTENNA_ID", antenna_id_table)

        beam_id_table = -1 * np.ones(nfeeds_table, dtype=np.int32)
        feed_table.putcol("BEAM_ID", beam_id_table)

        beam_offset_table = np.zeros((nfeeds_table, 2, 2), dtype=np.float64)
        feed_table.putcol("BEAM_OFFSET", beam_offset_table)

        feed_id_table = np.zeros(nfeeds_table, dtype=np.int32)
        feed_table.putcol("FEED_ID", feed_id_table)

        interval_table = np.zeros(nfeeds_table, dtype=np.float64)
        feed_table.putcol("INTERVAL", interval_table)

        # we want "x" or "y", *not* "e" or "n", so as not to confuse CASA
        pol_str = uvutils.polnum2str(self.polarization_array)
        feed_pols = {feed for pol in pol_str for feed in uvutils.POL_TO_FEED_DICT[pol]}
        nfeed_pols = len(feed_pols)
        num_receptors_table = np.tile(np.int32(nfeed_pols), nfeeds_table)
        feed_table.putcol("NUM_RECEPTORS", num_receptors_table)

        pol_types = [pol.upper() for pol in sorted(feed_pols)]
        pol_type_table = np.tile(pol_types, (nfeeds_table, 1))
        feed_table.putcol("POLARIZATION_TYPE", pol_type_table)

        pol_response_table = np.dstack(
            [np.eye(2, dtype=np.complex64) for n in range(nfeeds_table)]
        ).transpose()
        feed_table.putcol("POL_RESPONSE", pol_response_table)

        position_table = np.zeros((nfeeds_table, 3), dtype=np.float64)
        feed_table.putcol("POSITION", position_table)

        receptor_angle_table = np.zeros((nfeeds_table, nfeed_pols), dtype=np.float64)
        feed_table.putcol("RECEPTOR_ANGLE", receptor_angle_table)

        spectral_window_id_table = -1 * np.ones(nfeeds_table, dtype=np.int32)
        feed_table.putcol("SPECTRAL_WINDOW_ID", spectral_window_id_table)

        feed_table.done()

    def _write_ms_field(self, filepath):
        """
        Write out the field information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without FIELD suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        field_table = tables.table(filepath + "::FIELD", ack=False, readonly=False)
        time_val = (
            Time(np.median(self.time_array), format="jd", scale="utc").tai.mjd * 86400.0
        )
        n_poly = 0

        var_ref = False
        coord_frame = self.phase_center_frame
        coord_epoch = self.phase_center_epoch
        if self.multi_phase_center:
            for key in self.phase_center_catalog.keys():
                sou_frame = self.phase_center_catalog[key]["cat_frame"]
                sou_epoch = self.phase_center_catalog[key]["cat_epoch"]

                if (coord_frame != sou_frame) or (coord_epoch != sou_epoch):
                    var_ref = True
                    coord_frame = self.phase_center_frame
                    coord_epoch = self.phase_center_epoch

        if var_ref:
            var_ref_dict = {
                key: value for value, key in enumerate(COORD_UVDATA2CASA_DICT.keys())
            }
            col_ref_dict = {
                "PHASE_DIR": "PhaseDir_Ref",
                "DELAY_DIR": "DelayDir_Ref",
                "REFERENCE_DIR": "RefDir_Ref",
            }
            for key in col_ref_dict.keys():
                fieldcoldesc = tables.makearrcoldesc(
                    col_ref_dict[key],
                    0,
                    valuetype="int",
                    datamanagertype="StandardStMan",
                    datamanagergroup="field standard manager",
                )
                del fieldcoldesc["desc"]["shape"]
                del fieldcoldesc["desc"]["ndim"]
                del fieldcoldesc["desc"]["_c_order"]

                field_table.addcols(fieldcoldesc)
                field_table.getcolkeyword(key, "MEASINFO")

        if coord_frame is not None:
            ref_frame = self._parse_pyuvdata_frame_ref(coord_frame, coord_epoch)
            for col in ["PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"]:
                meas_info_dict = field_table.getcolkeyword(col, "MEASINFO")
                meas_info_dict["Ref"] = ref_frame
                if var_ref:
                    rev_ref_dict = {value: key for key, value in var_ref_dict.items()}
                    meas_info_dict["TabRefTypes"] = [
                        rev_ref_dict[key] for key in sorted(rev_ref_dict.keys())
                    ]
                    meas_info_dict["TabRefCodes"] = np.arange(
                        len(rev_ref_dict.keys()), dtype=np.int32
                    )
                    meas_info_dict["VarRefCol"] = col_ref_dict[col]

                field_table.putcolkeyword(col, "MEASINFO", meas_info_dict)

        if self.multi_phase_center:
            sou_list = list(self.phase_center_catalog.keys())
            sou_list.sort()
        else:
            sou_list = [
                self.object_name,
            ]

        for idx, key in enumerate(sou_list):
            if self.multi_phase_center:
                cat_dict = self.phase_center_catalog[key]
                phasedir = np.array([[cat_dict["cat_lon"], cat_dict["cat_lat"]]])
                sou_id = cat_dict["cat_id"]
                ref_dir = self._parse_pyuvdata_frame_ref(
                    cat_dict["cat_frame"], cat_dict["cat_epoch"], raise_error=var_ref,
                )
            else:
                phasedir = np.array([[self.phase_center_ra, self.phase_center_dec]])
                sou_id = 0
                assert self.phase_center_epoch == 2000.0

            field_table.addrows()

            field_table.putcell("DELAY_DIR", idx, phasedir)
            field_table.putcell("PHASE_DIR", idx, phasedir)
            field_table.putcell("REFERENCE_DIR", idx, phasedir)
            field_table.putcell("NAME", idx, key)
            field_table.putcell("NUM_POLY", idx, n_poly)
            field_table.putcell("TIME", idx, time_val)
            field_table.putcell("SOURCE_ID", idx, sou_id)
            if var_ref:
                for key in col_ref_dict.keys():
                    field_table.putcell(col_ref_dict[key], idx, var_ref_dict[ref_dir])

        field_table.done()

    def _write_ms_source(self, filepath):
        """
        Write out the source information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without SOURCE suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        source_desc = tables.complete_ms_desc("SOURCE")
        source_table = tables.table(
            filepath + "/SOURCE",
            tabledesc=source_desc,
            dminfo=tables.makedminfo(source_desc),
            ack=False,
            readonly=False,
        )

        time_val = (
            Time(np.median(self.time_array), format="jd", scale="utc").tai.mjd * 86400.0
        )
        int_val = np.finfo(float).max

        if self.multi_phase_center:
            sou_list = list(self.phase_center_catalog.keys())
        else:
            sou_list = [
                self.object_name,
            ]

        row_count = 0
        for sou_name in sou_list:
            if self.multi_phase_center:
                sou_ra = self.phase_center_catalog[sou_name]["cat_lon"]
                sou_dec = self.phase_center_catalog[sou_name]["cat_lat"]
                pm_ra = self.phase_center_catalog[sou_name].get("cat_pm_ra")
                pm_dec = self.phase_center_catalog[sou_name].get("cat_pm_dec")
                if (pm_ra is None) or (pm_dec is None):
                    pm_ra = 0.0
                    pm_dec = 0.0
                sou_id = self.phase_center_catalog[sou_name]["cat_id"]
            else:
                sou_ra = self.phase_center_ra
                sou_dec = self.phase_center_dec
                pm_ra = 0.0
                pm_dec = 0.0
                sou_id = 0

            sou_dir = np.array([sou_ra, sou_dec])
            pm_dir = np.array([pm_ra, pm_dec])
            for jdx in range(self.Nspws):
                source_table.addrows()
                source_table.putcell("SOURCE_ID", row_count, sou_id)
                source_table.putcell("TIME", row_count, time_val)
                source_table.putcell("INTERVAL", row_count, int_val)
                source_table.putcell("SPECTRAL_WINDOW_ID", row_count, jdx)
                source_table.putcell("NUM_LINES", row_count, 0)
                source_table.putcell("NAME", row_count, sou_name)
                source_table.putcell("DIRECTION", row_count, sou_dir)
                source_table.putcell("PROPER_MOTION", row_count, pm_dir)
                row_count += 1

        # We have one final thing we need to do here, which is to link the SOURCE table
        # to the main table, since it's not included in the list of default subtables.
        # You are _supposed_ to be able to provide a string, but it looks like that
        # functionality is actually broken in CASA. Fortunately, supplying the whole
        # table does seem to work (as least w/ casscore v3.3.1).
        ms = tables.table(filepath, readonly=False, ack=False)
        ms.putkeyword("SOURCE", source_table)
        ms.done()

        source_table.done()

    def _write_ms_pointing(self, filepath):
        """
        Write out the pointing information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without POINTING suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        pointing_table = tables.table(
            filepath + "::POINTING", ack=False, readonly=False
        )
        nants = np.max(self.antenna_numbers) + 1
        antennas = np.arange(nants, dtype=np.int32)

        # We want the number of unique timestamps here, since CASA wants a
        # per-timestamp, per-antenna entry
        times = np.asarray(
            [
                Time(t, format="jd", scale="utc").tai.mjd * 3600.0 * 24.0
                for t in np.unique(self.time_array)
            ]
        )

        # Same for the number of intervals
        intervals = np.array(
            [
                np.median(self.integration_time[self.time_array == timestamp])
                for timestamp in np.unique(self.time_array)
            ]
        )

        ntimes = len(times)

        antennas_full = np.tile(antennas, ntimes)
        times_full = np.repeat(times, nants)
        intervals_full = np.repeat(intervals, nants)

        nrows = len(antennas_full)
        assert len(times_full) == nrows
        assert len(intervals_full) == nrows

        # This extra step of adding a single row and plugging in values for DIRECTION
        # and TARGET are a workaround for a weird issue that pops up where, because the
        # values are not a fixed size (they are shape(2, Npoly), where Npoly is allowed
        # to vary from integration to integration), casacore seems to be very slow
        # filling in the data after the fact. By "pre-filling" the first row, the
        # addrows operation will automatically fill in an appropriately shaped array.
        # TODO: This works okay for steerable dishes, but less well for non-tracking
        # arrays (i.e., where primary beam center != phase center). Fix later.

        pointing_table.addrows(1)
        pointing_table.putcell("DIRECTION", 0, np.zeros((2, 1), dtype=np.float64))
        pointing_table.putcell("TARGET", 0, np.zeros((2, 1), dtype=np.float64))
        pointing_table.addrows(nrows - 1)
        pointing_table.done()

        pointing_table = tables.table(
            filepath + "::POINTING", ack=False, readonly=False
        )

        pointing_table.putcol("ANTENNA_ID", antennas_full, nrow=nrows)
        pointing_table.putcol("TIME", times_full, nrow=nrows)
        pointing_table.putcol("INTERVAL", intervals_full, nrow=nrows)

        name_col = np.asarray(["ZENITH"] * nrows, dtype=np.bytes_)
        pointing_table.putcol("NAME", name_col, nrow=nrows)

        num_poly_col = np.zeros(nrows, dtype=np.int32)
        pointing_table.putcol("NUM_POLY", num_poly_col, nrow=nrows)

        time_origin_col = times_full
        pointing_table.putcol("TIME_ORIGIN", time_origin_col, nrow=nrows)

        # we always "point" at zenith
        direction_col = np.zeros((nrows, 2, 1), dtype=np.float64)
        direction_col[:, 1, :] = np.pi / 2
        pointing_table.putcol("DIRECTION", direction_col, nrow=nrows)

        # just reuse "direction" for "target"
        pointing_table.putcol("TARGET", direction_col, nrow=nrows)

        # set tracking info to `False`
        tracking_col = np.zeros(nrows, dtype=np.bool_)
        pointing_table.putcol("TRACKING", tracking_col, nrow=nrows)

        pointing_table.done()

    def _write_ms_spectralwindow(self, filepath):
        """
        Write out the spectral information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without SPECTRAL_WINDOW suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        sw_table = tables.table(
            filepath + "::SPECTRAL_WINDOW", ack=False, readonly=False
        )

        if self.future_array_shapes:
            freq_array = self.freq_array
            ch_width = self.channel_width
        else:
            freq_array = self.freq_array[0]
            ch_width = np.zeros_like(freq_array) + self.channel_width

        spwidcoldesc = tables.makearrcoldesc(
            "ASSOC_SPW_ID",
            0,
            valuetype="int",
            ndim=-1,
            datamanagertype="StandardStMan",
            datamanagergroup="SpW optional column Standard Manager",
            comment="Associated spectral window id",
        )
        del spwidcoldesc["desc"]["shape"]
        sw_table.addcols(spwidcoldesc)

        spwassoccoldesc = tables.makearrcoldesc(
            "ASSOC_NATURE",
            "",
            valuetype="string",
            ndim=-1,
            datamanagertype="StandardStMan",
            datamanagergroup="SpW optional column Standard Manager",
            comment="Nature of association with other spectral window",
        )
        sw_table.addcols(spwassoccoldesc)

        for idx, spw_id in enumerate(self.spw_array):
            if self.flex_spw:
                ch_mask = self.flex_spw_id_array == spw_id
            else:
                ch_mask = np.ones(freq_array.shape, dtype=bool)
            sw_table.addrows()
            sw_table.putcell("NUM_CHAN", idx, np.sum(ch_mask))
            sw_table.putcell("NAME", idx, "SPW%d" % spw_id)
            sw_table.putcell("ASSOC_SPW_ID", idx, spw_id)
            sw_table.putcell("ASSOC_NATURE", idx, "")  # Blank for now
            sw_table.putcell("CHAN_FREQ", idx, freq_array[ch_mask])
            sw_table.putcell("CHAN_WIDTH", idx, ch_width[ch_mask])
            sw_table.putcell("EFFECTIVE_BW", idx, ch_width[ch_mask])
            sw_table.putcell("TOTAL_BANDWIDTH", idx, np.sum(ch_width[ch_mask]))
            sw_table.putcell("RESOLUTION", idx, ch_width[ch_mask])
            # TODO: These are placeholders for now, but should be replaced with
            # actual frequency reference info (once UVData handles that)
            sw_table.putcell("MEAS_FREQ_REF", idx, VEL_DICT["LSRK"])
            sw_table.putcell("REF_FREQUENCY", idx, freq_array[0])

        sw_table.done()

    def _write_ms_observation(self, filepath):
        """
        Write out the observation information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without OBSERVATION suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        observation_table = tables.table(
            filepath + "::OBSERVATION", ack=False, readonly=False
        )
        observation_table.addrows()
        observation_table.putcell("TELESCOPE_NAME", 0, self.telescope_name)

        # It appears that measurement sets do not have a concept of a telescope location
        # We add it here as a non-standard column in order to round trip it properly
        name_col_desc = tableutil.makearrcoldesc(
            "TELESCOPE_LOCATION",
            self.telescope_location[0],
            shape=[3],
            valuetype="double",
        )
        observation_table.addcols(name_col_desc)
        observation_table.putcell("TELESCOPE_LOCATION", 0, self.telescope_location)

        extra_upper = [key.upper() for key in self.extra_keywords.keys()]
        if "OBSERVER" in extra_upper:
            key_ind = extra_upper.index("OBSERVER")
            key = list(self.extra_keywords.keys())[key_ind]
            observation_table.putcell("OBSERVER", 0, self.extra_keywords[key])
        else:
            observation_table.putcell("OBSERVER", 0, self.telescope_name)

        observation_table.done()

    def _write_ms_polarization(self, filepath):
        """
        Write out the polarization information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without POLARIZATION suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        pol_str = uvutils.polnum2str(self.polarization_array)
        feed_pols = {feed for pol in pol_str for feed in uvutils.POL_TO_FEED_DICT[pol]}
        pol_types = [pol.lower() for pol in sorted(feed_pols)]
        pol_tuples = np.asarray(
            [(pol_types.index(i), pol_types.index(j)) for i, j in pol_str],
            dtype=np.int32,
        )

        pol_table = tables.table(filepath + "::POLARIZATION", ack=False, readonly=False)
        pol_table.addrows()
        pol_table.putcell(
            "CORR_TYPE",
            0,
            np.array(
                [POL_AIPS2CASA_DICT[aipspol] for aipspol in self.polarization_array]
            ),
        )
        pol_table.putcell("CORR_PRODUCT", 0, pol_tuples)
        pol_table.putcell("NUM_CORR", 0, self.Npols)

        pol_table.done()

    def _ms_hist_to_string(self, history_table):
        """
        Convert a CASA history table into a string for the uvdata history parameter.

        Also stores messages column as a list for consitency with other uvdata types

        Parameters
        ----------
        history_table : a casa table object
            CASA table with history information.

        Returns
        -------
        str
            string enconding complete casa history table converted with a new
            line denoting rows and a ';' denoting column breaks.
        pyuvdata_written :  bool
            boolean indicating whether or not this MS was written by pyuvdata.

        """
        # string to store all the casa history info
        history_str = ""
        pyuvdata_written = False

        # Do not touch the history table if it has no information
        if history_table.nrows() > 0:
            history_str = (
                "Begin measurement set history\n"
                "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE;"
                "OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n"
            )

            app_params = history_table.getcol("APP_PARAMS")["array"]
            # TODO: might need to handle the case where cli_command is empty
            cli_command = history_table.getcol("CLI_COMMAND")["array"]
            application = history_table.getcol("APPLICATION")
            message = history_table.getcol("MESSAGE")
            obj_id = history_table.getcol("OBJECT_ID")
            obs_id = history_table.getcol("OBSERVATION_ID")
            origin = history_table.getcol("ORIGIN")
            priority = history_table.getcol("PRIORITY")
            times = history_table.getcol("TIME")
            # Now loop through columns and generate history string
            ntimes = len(times)
            cols = [
                app_params,
                cli_command,
                application,
                message,
                obj_id,
                obs_id,
                origin,
                priority,
                times,
            ]

            # if this MS was written by pyuvdata, some history that originated in
            # pyuvdata is in the MS history table. We separate that out since it doesn't
            # really belong to the MS history block (and so round tripping works)
            pyuvdata_line_nos = [
                line_no
                for line_no, line in enumerate(application)
                if "pyuvdata" in line
            ]

            for tbrow in range(ntimes):
                # first check to see if this row was put in by pyuvdata.
                # If so, don't mix them in with the MS stuff
                if tbrow in pyuvdata_line_nos:
                    continue

                newline = ";".join([str(col[tbrow]) for col in cols]) + "\n"
                history_str += newline

            history_str += "End measurement set history.\n"

            # make a list of lines that are MS specific (i.e. not pyuvdata lines)
            ms_line_nos = np.arange(ntimes).tolist()
            for count, line_no in enumerate(pyuvdata_line_nos):
                # Subtract off the count, since the line_no will effectively
                # change with each call to pop on the list
                ms_line_nos.pop(line_no - count)

            # Handle the case where there is no history but pyuvdata
            if len(ms_line_nos) == 0:
                ms_line_nos = [-1]

            if len(pyuvdata_line_nos) > 0:
                pyuvdata_written = True
                for line_no in pyuvdata_line_nos:
                    if line_no < min(ms_line_nos):
                        # prepend these lines to the history
                        history_str = message[line_no] + "\n" + history_str
                    else:
                        # append these lines to the history
                        history_str += message[line_no] + "\n"

        return history_str, pyuvdata_written

    def _write_ms_history(self, filepath):
        """
        Parse the history into an MS history table.

        If the history string contains output from `_ms_hist_to_string`, parse that back
        into the MS history table.

        Parameters
        ----------
        filepath : str
            path to MS (without HISTORY suffix)
        history : str
            A history string that may or may not contain output from
            `_ms_hist_to_string`.

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        app_params = []
        cli_command = []
        application = []
        message = []
        obj_id = []
        obs_id = []
        origin = []
        priority = []
        times = []
        if "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE" in self.history:
            # this history contains info from an MS history table. Need to parse it.

            ms_header_line_no = None
            ms_end_line_no = None
            pre_ms_history_lines = []
            post_ms_history_lines = []
            for line_no, line in enumerate(self.history.splitlines()):
                if "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE" in line:
                    ms_header_line_no = line_no
                    # we don't need this line anywhere below so continue
                    continue

                if "End measurement set history" in line:
                    ms_end_line_no = line_no
                    # we don't need this line anywhere below so continue
                    continue

                if ms_header_line_no is not None and ms_end_line_no is None:

                    # this is part of the MS history block. Parse it.
                    line_parts = line.split(";")

                    app_params.append(line_parts[0])
                    cli_command.append(line_parts[1])
                    application.append(line_parts[2])
                    message.append(line_parts[3])
                    obj_id.append(int(line_parts[4]))
                    obs_id.append(int(line_parts[5]))
                    origin.append(line_parts[6])
                    priority.append(line_parts[7])
                    times.append(np.float64(line_parts[8]))
                elif ms_header_line_no is None:
                    # this is before the MS block
                    if "Begin measurement set history" not in line:
                        pre_ms_history_lines.append(line)
                else:
                    # this is after the MS block
                    post_ms_history_lines.append(line)

            for line_no, line in enumerate(pre_ms_history_lines):
                app_params.insert(line_no, "")
                cli_command.insert(line_no, "")
                application.insert(line_no, "pyuvdata")
                message.insert(line_no, line)
                obj_id.insert(line_no, 0)
                obs_id.insert(line_no, -1)
                origin.insert(line_no, "pyuvdata")
                priority.insert(line_no, "INFO")
                times.insert(line_no, Time.now().mjd * 3600.0 * 24.0)

            for line_no, line in enumerate(post_ms_history_lines):
                app_params.append("")
                cli_command.append("")
                application.append("pyuvdata")
                message.append(line)
                obj_id.append(0)
                obs_id.append(-1)
                origin.append("pyuvdata")
                priority.append("INFO")
                times.append(Time.now().mjd * 3600.0 * 24.0)

        else:
            # no prior MS history detected in the history. Put all of our history in
            # the message column
            for line in self.history.splitlines():
                app_params.append("")
                cli_command.append("")
                application.append("pyuvdata")
                message.append(line)
                obj_id.append(0)
                obs_id.append(-1)
                origin.append("pyuvdata")
                priority.append("INFO")
                times.append(Time.now().mjd * 3600.0 * 24.0)

        history_table = tables.table(filepath + "::HISTORY", ack=False, readonly=False)

        nrows = len(message)
        history_table.addrows(nrows)

        # the first two lines below break on python-casacore < 3.1.0
        history_table.putcol("APP_PARAMS", np.asarray(app_params)[:, np.newaxis])
        history_table.putcol("CLI_COMMAND", np.asarray(cli_command)[:, np.newaxis])
        history_table.putcol("APPLICATION", application)
        history_table.putcol("MESSAGE", message)
        history_table.putcol("OBJECT_ID", obj_id)
        history_table.putcol("OBSERVATION_ID", obs_id)
        history_table.putcol("ORIGIN", origin)
        history_table.putcol("PRIORITY", priority)
        history_table.putcol("TIME", times)

        history_table.done()

    def write_ms(
        self,
        filepath,
        force_phase=False,
        clobber=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Write a CASA measurement set (MS).

        Parameters
        ----------
        filepath : str
            The MS file path to write to.
        force_phase : bool
            Option to automatically phase drift scan data to zenith of the first
            timestamp.
        clobber : bool
            Option to overwrite the file if it already exists.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if os.path.exists(filepath):
            if clobber:
                print("File exists; clobbering")
            else:
                raise IOError("File exists; skipping")

        # CASA does not have a way to handle "unphased" data in the way that UVData
        # objects can, so we need to check here whether or not any such data exists
        # (and if need be, fix it).
        if np.any(self._check_for_unphased()):
            if force_phase:
                print(
                    "The data are in drift mode and do not have a "
                    "defined phase center. Phasing to zenith of the first "
                    "timestamp."
                )
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(
                    phase_time,
                    select_mask=self._check_for_unphased()
                    if self.multi_phase_center
                    else None,
                )
            else:
                raise ValueError(
                    "The data are in drift mode. "
                    "Set force_phase to true to phase the data "
                    "to zenith of the first timestamp before "
                    "writing a measurement set file."
                )

        # If scan numbers are not already defined from reading an MS,
        # group integrations (rows) into scan numbers.
        if self.scan_number_array is None:
            self._set_scan_numbers()

        # Initialize a skelton measurement set
        ms = self._init_ms_file(filepath)

        attr_list = ["data_array", "nsample_array", "flag_array"]
        col_list = ["DATA", "WEIGHT_SPECTRUM", "FLAG"]

        # Some tasks in CASA require a band-representative (band-averaged?) value for
        # the weights and noise for all channels in each row in the MAIN table, which
        # we will roughly calculate in temp_weights below.
        temp_weights = np.zeros((self.Nblts * self.Nspws, self.Npols), dtype=float)
        data_desc_array = np.zeros((self.Nblts * self.Nspws))

        # astropys Time has some overheads associated with it, so use unique to run
        # this date conversion as few times as possible. Note that the default for MS
        # is MJD UTC seconds, versus JD UTC days for UVData.
        time_array, time_ind = np.unique(self.time_array, return_inverse=True)
        time_array = (Time(time_array, format="jd").mjd * 3600.0 * 24.0)[time_ind]

        # Add all the rows we need up front, which will allow us to fill the
        # columns all in one shot.
        ms.addrows(self.Nblts * self.Nspws)
        if self.Nspws == 1:
            # If we only have one spectral window, there is nothing we need to worry
            # about ordering, so just write the data-related arrays as is to disk
            for attr, col in zip(attr_list, col_list):
                if self.future_array_shapes:
                    ms.putcol(col, getattr(self, attr))
                else:
                    ms.putcol(col, np.squeeze(getattr(self, attr), axis=1))

            # Band-averaged weights are used for some things in CASA - calculate them
            # here using median nsamples.
            if self.future_array_shapes:
                temp_weights = np.median(self.nsample_array, axis=1)
            else:
                temp_weights = np.squeeze(np.median(self.nsample_array, axis=2), axis=1)

            # Grab pointers for the per-blt record arrays
            ant_1_array = self.ant_1_array
            ant_2_array = self.ant_2_array
            integration_time = self.integration_time
            uvw_array = self.uvw_array
            scan_number_array = self.scan_number_array
        else:
            # If we have _more_ than one spectral window, then we need to handle each
            # window separately, since they can have differing numbers of channels.
            # (n.b., tables.putvarcol can write complex tables like these, but its
            # slower and more memory-intensive than putcol).

            # Since muliple records trace back to a single baseline-time, we use this
            # array to map from arrays that store on a per-record basis to positions
            # within arrays that record metadata on a per-blt basis.
            blt_map_array = np.zeros((self.Nblts * self.Nspws), dtype=int)

            # we will select out individual spectral windows several times, so create
            # these masks once up front before we enter the loop.
            spw_sel_dict = {}
            for spw_id in self.spw_array:
                spw_sel_dict[spw_id] = self.flex_spw_id_array == spw_id

            # Based on some analysis of ALMA/ACA data, various routines in CASA appear
            # to prefer data be grouped together on a "per-scan" basis, then per-spw,
            # and then the more usual selections of per-time, per-ant1, etc.
            last_row = 0
            for scan_num in sorted(np.unique(self.scan_number_array)):
                # Select all data from the scan
                scan_screen = self.scan_number_array == scan_num

                # Get the number of records inside the scan, where 1 record = 1 spw in
                # 1 baseline at 1 time
                Nrecs = np.sum(scan_screen)

                # Record which SPW/"Data Description" this data is matched to
                data_desc_array[last_row : last_row + (Nrecs * self.Nspws)] = np.repeat(
                    np.arange(self.Nspws), Nrecs,
                )

                # Record index positions
                blt_map_array[last_row : last_row + (Nrecs * self.Nspws)] = np.tile(
                    np.where(scan_screen)[0], self.Nspws,
                )

                # Extract out the relevant data out of our data-like arrays that
                # belong to this scan number.
                val_dict = {}
                for attr, col in zip(attr_list, col_list):
                    if self.future_array_shapes:
                        val_dict[col] = getattr(self, attr)[scan_screen]
                    else:
                        val_dict[col] = np.squeeze(
                            getattr(self, attr)[scan_screen], axis=1
                        )

                # This is where the bulk of the heavy lifting is - use the per-spw
                # channel masks to record one spectral window at a time.
                for idx, key in enumerate(self.spw_array):
                    for col in col_list:
                        ms.putcol(
                            col, val_dict[col][:, spw_sel_dict[key]], last_row, Nrecs
                        )

                    # Tally here the "wideband" weights for the whole spectral window,
                    # which is used in some CASA routines.
                    temp_weights[last_row : last_row + Nrecs] = np.median(
                        val_dict["WEIGHT_SPECTRUM"][:, spw_sel_dict[key]], axis=1
                    )
                    last_row += Nrecs

            # Now that we have an array to map baselime-time to individual records,
            # use our indexing array to map various metadata.
            ant_1_array = self.ant_1_array[blt_map_array]
            ant_2_array = self.ant_2_array[blt_map_array]
            integration_time = self.integration_time[blt_map_array]
            time_array = time_array[blt_map_array]
            uvw_array = self.uvw_array[blt_map_array]
            scan_number_array = self.scan_number_array[blt_map_array]

        # Write out the units of the visibilities, post a warning if its not in Jy since
        # we don't know how every CASA program may react
        ms.putcolkeyword("DATA", "QuantumUnits", self.vis_units)
        if self.vis_units != "Jy":
            warnings.warn(
                "Writing in the MS file that the units of the data are %s, although "
                "some CASA process will ignore this and assume the units are all in Jy "
                "(or may not know how to handle data in these units)." % self.vis_units
            )

        # TODO: If/when UVData objects can store visibility noise estimates, update
        # the code below to capture those.
        ms.putcol("WEIGHT", temp_weights)
        ms.putcol("SIGMA", np.power(temp_weights, -0.5, where=temp_weights != 0))

        ms.putcol("ANTENNA1", ant_1_array)
        ms.putcol("ANTENNA2", ant_2_array)

        # "INVERVAL" refers to "width" of the window of time time over which data was
        # collected, while "EXPOSURE" is the sum total of integration time.  UVData
        # does not differentiate between these concepts, hence why one array is used
        # for both values.
        ms.putcol("INTERVAL", integration_time)
        ms.putcol("EXPOSURE", integration_time)

        ms.putcol("DATA_DESC_ID", data_desc_array)
        ms.putcol("SCAN_NUMBER", scan_number_array)
        ms.putcol("TIME", time_array)
        ms.putcol("TIME_CENTROID", time_array)

        # FITS uvw direction convention is opposite ours and Miriad's.
        # CASA's convention is unclear: the docs contradict themselves,
        # but after a question to the helpdesk we got a clear response that
        # the convention is antenna2_pos - antenna1_pos, so the convention is the
        # same as ours & Miriad's
        ms.putcol("UVW", uvw_array)

        if self.multi_phase_center:
            # We have to do an extra bit of work here, as CASA won't accept arbitrary
            # values for field ID (rather, the ID number matches to the row number in
            # the FIELD subtable). When we write out the fields, we use sort so that
            # we can reproduce the same ordering here.
            field_ids = np.empty_like(self.phase_center_id_array)
            sou_list = list(self.phase_center_catalog.keys())
            sou_list.sort()
            for idx in range(self.Nphase):
                sel_mask = np.equal(
                    self.phase_center_id_array,
                    self.phase_center_catalog[sou_list[idx]]["cat_id"],
                )

                field_ids[sel_mask] = idx

            ms.putcol(
                "FIELD_ID", field_ids[blt_map_array] if self.Nspws > 1 else field_ids
            )

        # Finally, record extra keywords and x_orientation, both of which the MS format
        # doesn't quite have equivalent fields to stuff data into (and instead is put
        # into the main header as a keyword).
        if len(self.extra_keywords) != 0:
            ms.putkeyword("pyuvdata_extra", self.extra_keywords)

        if self.x_orientation is not None:
            ms.putkeyword("pyuvdata_xorient", self.x_orientation)

        ms.done()

        self._write_ms_antenna(filepath)
        self._write_ms_data_description(filepath)
        self._write_ms_feed(filepath)
        self._write_ms_field(filepath)
        self._write_ms_source(filepath)
        self._write_ms_spectralwindow(filepath)
        self._write_ms_pointing(filepath)
        self._write_ms_polarization(filepath)
        self._write_ms_observation(filepath)
        self._write_ms_history(filepath)

    def _parse_casa_frame_ref(self, ref_name, raise_error=True):
        """
        Interpret a CASA frame into an astropy-friendly frame and epoch.

        Parameters
        ----------
        ref_name : str
            Name of the CASA-based spatial coordinate reference frame.
        raise_error : bool
            Whether to raise an error if the name has no match. Default is True, if set
            to false will raise a warning instead.

        Returns
        -------
        frame_name : str
            Name of the frame. Typically matched to the UVData attribute
            `phase_center_frame`.
        epoch_val : float
            Epoch value for the given frame, in Julian years unless `frame_name="FK4"`,
            in which case the value is in Besselian years. Typically matched to the
            UVData attribute `phase_center_epoch`.

        Raises
        ------
        ValueError
            If the CASA coordinate frame does not match the known supported list of
            frames for CASA.
        NotImplementedError
            If using a CASA coordinate frame that does not yet have a corresponding
            astropy frame that is supported by pyuvdata.
        """
        frame_name = None
        epoch_val = None
        try:
            frame_tuple = COORD_UVDATA2CASA_DICT[ref_name]
            if frame_tuple is None:
                if raise_error:
                    raise NotImplementedError(
                        "Support for the %s frame is not yet supported." % ref_name
                    )
                else:
                    warnings.warn(
                        "Support for the %s frame is not yet supported." % ref_name
                    )
            else:
                frame_name = frame_tuple[0]
                epoch_val = frame_tuple[1]
        except KeyError:
            if raise_error:
                raise ValueError(
                    "The coordinate frame %s is not one of the supported frames for "
                    "measurement sets." % ref_name
                )
            else:
                warnings.warn(
                    "The coordinate frame %s is not one of the supported frames for "
                    "measurement sets." % ref_name
                )

        return frame_name, epoch_val

    def _parse_pyuvdata_frame_ref(self, frame_name, epoch_val, raise_error=True):
        """
        Interpret a UVData pair of frame + epoch into a CASA frame name.

        Parameters
        ----------
        frame_name : str
            Name of the frame. Typically matched to the UVData attribute
            `phase_center_frame`.
        epoch_val : float
            Epoch value for the given frame, in Julian years unless `frame_name="FK4"`,
            in which case the value is in Besselian years. Typically matched to the
            UVData attribute `phase_center_epoch`.
        raise_error : bool
            Whether to raise an error if the name has no match. Default is True, if set
            to false will raise a warning instead.

        Returns
        -------
        ref_name : str
            Name of the CASA-based spatial coordinate reference frame.

        Raises
        ------
        ValueError
            If the provided coordinate frame and/or epoch value has no matching
            counterpart to those supported in CASA.

        """
        # N.B. -- this is something of a stub for a more sophisticated function to
        # handle this. For now, this just does a reverse lookup on the limited frames
        # supported by UVData objects, although eventually it can be expanded to support
        # more native MS frames.
        reverse_dict = {ref: key for key, ref in COORD_UVDATA2CASA_DICT.items()}

        ref_name = None
        try:
            ref_name = reverse_dict[(str(frame_name), float(epoch_val))]
        except KeyError:
            if raise_error:
                raise ValueError(
                    "Frame %s (epoch %g) does not have a corresponding match to "
                    "supported frames in the MS file format." % (frame_name, epoch_val)
                )
            else:
                warnings.warn(
                    "Frame %s (epoch %g) does not have a corresponding match to "
                    "supported frames in the MS file format." % (frame_name, epoch_val)
                )

        return ref_name

    def _read_ms_main(
        self,
        filepath,
        data_column,
        data_desc_dict,
        read_weights=True,
        flip_conj=False,
        raise_error=True,
    ):
        """
        Read data from the main table of a MS file.

        This method is not meant to be called by users, and is instead a utility
        function for the `read_ms` method (which users should call instead).

        Parameters
        ----------
        filepath : str
            The measurement set root directory to read from.
        data_column : str
            name of CASA data column to read into data_array. Options are:
            'DATA', 'MODEL', or 'CORRECTED_DATA'
        data_desc_dict : dict
            Dictionary describing the various rows in the DATA_DESCRIPTION table of
            an MS file. Keys match to the individual rows, and the values are themselves
            dicts containing several keys (including "CORR_TYPE", "SPW_ID", "NUM_CORR",
            "NUM_CHAN").
        read_weights : bool
            Read in the weights from the MS file, default is True. If false, the method
            will set the `nsamples_array` to the same uniform value (namely 1.0).
        flip_conj : bool
            On read, whether to flip the conjugation of the baselines. Normally the MS
            format is the same as what is used for pyuvdata (ant2 - ant1), hence the
            default is False.
        raise_error : bool
            On read, whether to raise an error if different records (i.e.,
            different spectral windows) report different metadata for the same
            time-baseline combination, which CASA allows by UVData does not. Default
            is True, if set to False will raise a warning instead.

        Returns
        -------
        spw_list : list of int
            List of SPW numbers present in the data set, equivalent to the attribute
            `spw_array` in a UVData object.
        field_list : list of int
            List of field IDs present in the data set. Matched to rows in the FIELD
            table for the measurement set.
        pol_list : list of int
            List of polarization IDs (in the AIPS convention) present in the data set.
            Equivalent to the attribute `polarization_array` in a UVData object.

        Raises
        ------
        ValueError
            If the `data_column` is not set to an allowed value.
            If the MS file contains data from multiple subarrays.
        """
        tb_main = tables.table(filepath, ack=False)

        main_keywords = tb_main.getkeywords()
        if "pyuvdata_extra" in main_keywords.keys():
            self.extra_keywords = main_keywords["pyuvdata_extra"]

        if "pyuvdata_xorient" in main_keywords.keys():
            self.x_orientation = main_keywords["pyuvdata_xorient"]

        default_vis_units = {"DATA": "uncalib", "CORRECTED_DATA": "Jy", "MODEL": "Jy"}

        # make sure user requests a valid data_column
        if data_column not in default_vis_units.keys():
            raise ValueError(
                "Invalid data_column value supplied. Use 'Data','MODEL' or"
                " 'CORRECTED_DATA'"
            )

        # set visibility units
        try:
            self.vis_units = tb_main.getcolkeywords(data_column)["QuantumUnits"]
        except KeyError:
            self.vis_units = default_vis_units[data_column]

        # limit length of extra_keywords keys to 8 characters to match uvfits & miriad
        self.extra_keywords["DATA_COL"] = data_column

        time_arr = tb_main.getcol("TIME")
        # N.b., EXPOSURE is what's needed for noise calculation, but INTERVAL defines
        # the time period over which the data are collected
        int_arr = tb_main.getcol("EXPOSURE")
        ant_1_arr = tb_main.getcol("ANTENNA1")
        ant_2_arr = tb_main.getcol("ANTENNA2")
        field_arr = tb_main.getcol("FIELD_ID")
        scan_number_arr = tb_main.getcol("SCAN_NUMBER")
        uvw_arr = tb_main.getcol("UVW")
        data_desc_arr = tb_main.getcol("DATA_DESC_ID")
        subarr_arr = tb_main.getcol("ARRAY_ID")
        unique_data_desc = np.unique(data_desc_arr)

        if len(np.unique(subarr_arr)) > 1:
            raise ValueError(
                "This file appears to have multiple subarray "
                "values; only files with one subarray are "
                "supported."
            )

        data_desc_count = np.sum(np.isin(list(data_desc_dict.keys()), unique_data_desc))

        if data_desc_count == 0:
            # If there are no records selected, then there isnt a whole lot to do
            return None, None, None
        elif data_desc_count == 1:
            # If we only have a single spectral window, then we can bypass a whole lot
            # of slicing and dicing on account of there being a one-to-one releationship
            # in rows of the MS to the per-blt records of UVData objects.
            self.time_array = Time(time_arr / 86400.0, format="mjd").jd
            self.integration_time = int_arr
            self.ant_1_array = ant_1_arr
            self.ant_2_array = ant_2_arr
            self.uvw_array = uvw_arr * ((-1) ** flip_conj)
            self.phase_center_id_array = field_arr
            self.scan_number_array = scan_number_arr

            self.flag_array = tb_main.getcol("FLAG")

            if flip_conj:
                self.data_array = np.conj(tb_main.getcol(data_column))
            else:
                self.data_array = tb_main.getcol(data_column)

            if read_weights:
                try:
                    self.nsample_array = tb_main.getcol("WEIGHT_SPECTRUM")
                except RuntimeError:
                    self.nsample_array = np.repeat(
                        np.expand_dims(tb_main.getcol("WEIGHT"), axis=1),
                        self.data_array.shape[1],
                        axis=1,
                    )
            else:
                self.nsample_array = np.ones_like(self.data_array, dtype=float)

            data_desc_key = np.intersect1d(
                unique_data_desc, list(data_desc_dict.keys())
            )[0]
            spw_list = [data_desc_dict[data_desc_key]["SPW_ID"]]
            self.flex_spw_id_array = spw_list[0] + np.zeros(
                data_desc_dict[data_desc_key]["NUM_CHAN"], dtype=int
            )

            field_list = np.unique(field_arr)
            pol_list = [
                POL_CASA2AIPS_DICT[key]
                for key in data_desc_dict[data_desc_key]["CORR_TYPE"]
            ]

            tb_main.close()
            return spw_list, field_list, pol_list

        tb_main.close()

        # If you are at this point, it means that we potentially have multiple spectral
        # windows to deal with, and so some additional care is required since MS does
        # NOT require data from all windows to be present simultaneously.

        use_row = np.zeros_like(time_arr, dtype=bool)
        data_dict = {}
        for key in data_desc_dict.keys():
            sel_mask = data_desc_arr == key

            if not np.any(sel_mask):
                continue

            use_row[sel_mask] = True
            data_dict[key] = dict(data_desc_dict[key])
            data_dict[key]["TIME"] = time_arr[sel_mask]  # Midpoint time in mjd seconds
            data_dict[key]["EXPOSURE"] = int_arr[sel_mask]  # Int time in sec
            data_dict[key]["ANTENNA1"] = ant_1_arr[sel_mask]  # First antenna
            data_dict[key]["ANTENNA2"] = ant_2_arr[sel_mask]  # Second antenna
            data_dict[key]["FIELD_ID"] = field_arr[sel_mask]  # Source ID
            data_dict[key]["SCAN_NUMBER"] = scan_number_arr[sel_mask]  # Scan number
            data_dict[key]["UVW"] = uvw_arr[sel_mask]  # UVW coords

        time_arr = time_arr[use_row]
        ant_1_arr = ant_1_arr[use_row]
        ant_2_arr = ant_2_arr[use_row]

        unique_blts = sorted(
            {
                (time, ant1, ant2)
                for time, ant1, ant2 in zip(time_arr, ant_1_arr, ant_2_arr)
            }
        )

        blt_dict = {}
        for idx, blt_tuple in enumerate(unique_blts):
            blt_dict[blt_tuple] = idx

        nblts = len(unique_blts)
        pol_list = np.unique([data_dict[key]["CORR_TYPE"] for key in data_dict.keys()])
        npols = len(pol_list)

        spw_list = {
            data_dict[key]["SPW_ID"]: (key, data_dict[key]["NUM_CHAN"])
            for key in data_dict.keys()
        }

        # Here we sort out where the various spectral windows are starting and stoping
        # in our flex_spw spectrum, if applicable. By default, data are sorted in
        # spw-number order.
        nfreqs = 0
        spw_id_array = np.array([], dtype=int)
        for key in sorted(spw_list.keys()):
            data_dict_key = spw_list[key][0]
            nchan = spw_list[key][1]
            data_dict[data_dict_key]["STARTCHAN"] = nfreqs
            data_dict[data_dict_key]["STOPCHAN"] = nfreqs + nchan
            data_dict[data_dict_key]["NCHAN"] = nchan
            spw_id_array = np.append(spw_id_array, [key] * nchan)
            nfreqs += nchan

        for key in sorted(data_dict.keys()):
            blt_idx = [
                blt_dict[(time, ant1, ant2)]
                for time, ant1, ant2 in zip(
                    data_dict[key]["TIME"],
                    data_dict[key]["ANTENNA1"],
                    data_dict[key]["ANTENNA2"],
                )
            ]

            data_dict[key]["BLT_IDX"] = np.array(blt_idx, dtype=int)
            data_dict[key]["NBLTS"] = len(blt_idx)

            pol_idx = np.intersect1d(
                pol_list,
                data_desc_dict[key]["CORR_TYPE"],
                assume_unique=True,
                return_indices=True,
            )[1]
            data_dict[key]["POL_IDX"] = pol_idx.astype(int)

        # We have all of the meta-information linked the various data desc IDs,
        # so now we can finally get to the business of filling in the actual data.
        data_array = np.zeros((nblts, nfreqs, npols), dtype=complex)
        nsample_array = np.ones((nblts, nfreqs, npols))
        flag_array = np.ones((nblts, nfreqs, npols), dtype=bool)

        # We will also fill in our own metadata on a per-blt basis here
        time_arr = np.zeros(nblts)
        int_arr = np.zeros(nblts)
        ant_1_arr = np.zeros(nblts, dtype=int)
        ant_2_arr = np.zeros(nblts, dtype=int)
        field_arr = np.zeros(nblts, dtype=int)
        scan_number_arr = np.zeros(nblts, dtype=int)
        uvw_arr = np.zeros((nblts, 3))

        # Since each data description (i.e., spectral window) record can technically
        # have its own values for time, int-time, etc, we want to check and verify that
        # the values are consistent on a per-blt basis (since that's the most granular
        # pyuvdata can store that information).
        has_data = np.zeros(nblts, dtype=bool)

        arr_tuple = (
            time_arr,
            int_arr,
            ant_1_arr,
            ant_2_arr,
            field_arr,
            scan_number_arr,
            uvw_arr,
        )
        name_tuple = (
            "TIME",
            "EXPOSURE",
            "ANTENNA1",
            "ANTENNA2",
            "FIELD_ID",
            "SCAN_NUMBER",
            "UVW",
        )
        vary_list = []
        for key in data_dict.keys():
            # Get the indexing information for the data array
            blt_idx = data_dict[key]["BLT_IDX"]
            startchan = data_dict[key]["STARTCHAN"]
            stopchan = data_dict[key]["STOPCHAN"]
            pol_idx = data_dict[key]["POL_IDX"]

            # Identify which values have already been populated with data, so we know
            # which values to check.
            check_mask = has_data[blt_idx]
            check_idx = blt_idx[check_mask]

            # Loop through the metadata fields we intend to populate
            for arr, name in zip(arr_tuple, name_tuple):
                if not np.allclose(data_dict[key][name][check_mask], arr[check_idx]):
                    if raise_error:
                        raise ValueError(
                            "Column %s appears to vary on between windows, which is "
                            "not permitted for UVData objects. To bypass this error, "
                            "you can set raise_error=False, which will raise a warning "
                            "instead and use the first recorded value." % name
                        )
                    elif name not in vary_list:
                        # If not raising an error, then at least warn the user that
                        # discrepant data were detected.
                        warnings.warn(
                            "Column %s appears to vary on between windows, defaulting "
                            "to first recorded value." % name
                        )
                        # Add to a list so we don't spew multiple warnings for one
                        # column (which could just flood the terminal).
                        vary_list.append(name)

                arr[blt_idx[~check_mask]] = data_dict[key][name][~check_mask]

            # Can has data now please?
            has_data[blt_idx] = True

            # Remove a slice out of the larger arrays for us to populate with an MS read
            # operation. This has the advantage that if different data descrips contain
            # different polarizations (which is allowed), it will populate the arrays
            # correctly, although for most files (where all pols are written in one
            # data descrip), this shouldn't matter.
            temp_data = data_array[blt_idx, startchan:stopchan]
            temp_flags = flag_array[blt_idx, startchan:stopchan]
            if read_weights:
                temp_weights = nsample_array[blt_idx, startchan:stopchan]

            # This TaQL call allows the low-level C++ routines to handle mapping data
            # access, and returns a table object that _only_ has records matching our
            # request. This allows one to do a simple and fast getcol for reading the
            # data, flags, and weights, since they should all be the same shape on a
            # per-row basis for the same data description. Alternative read methods
            # w/ getcell, getvarcol, and per-row getcols produced way slower code.
            tb_main_sel = tables.taql(
                "select from %s where DATA_DESC_ID == %i" % (filepath, key)
            )

            # Fill in the temp arrays, and then plug them back into the main array.
            # Note that this operation has to be split in two because you can only use
            # advanced slicing on one axis (which both blt_idx and pol_idx require).
            if flip_conj:
                temp_data[:, :, pol_idx] = np.conj(tb_main_sel.getcol("DATA"))
            else:
                temp_data[:, :, pol_idx] = tb_main_sel.getcol("DATA")

            temp_flags[:, :, pol_idx] = tb_main_sel.getcol("FLAG")

            data_array[blt_idx, startchan:stopchan] = temp_data
            flag_array[blt_idx, startchan:stopchan] = temp_flags

            if read_weights:
                # The weights can be stored in a couple of different columns, but we
                # use a try/except here to capture two separate cases (that both will
                # produce runtime errors) -- when WEIGHT_SPECTRUM isn't a column, and
                # when it is BUT its unfilled (which causes getcol to throw an error).
                try:
                    temp_weights[:, :, pol_idx] = tb_main_sel.getcol("WEIGHT_SPECTRUM")
                except RuntimeError:
                    temp_weights[:, :, pol_idx] = np.repeat(
                        np.expand_dims(tb_main_sel.getcol("WEIGHT"), axis=1),
                        data_desc_dict[key]["NUM_CHAN"],
                        axis=1,
                    )
                nsample_array[blt_idx, startchan:stopchan, :] = temp_weights

            # Close the table, get ready for the next loop
            tb_main_sel.close()

        self.data_array = data_array
        self.flag_array = flag_array
        self.nsample_array = nsample_array

        self.ant_1_array = ant_1_arr
        self.ant_2_array = ant_2_arr
        self.time_array = Time(time_arr / 86400.0, format="mjd").jd
        self.integration_time = int_arr
        self.uvw_array = uvw_arr * ((-1) ** flip_conj)
        self.phase_center_id_array = field_arr
        self.phase_center_id_array = field_arr
        self.scan_number_array = scan_number_arr
        self.flex_spw_id_array = spw_id_array

        pol_list = [POL_CASA2AIPS_DICT[key] for key in pol_list]
        spw_list = sorted(spw_list.keys())
        field_list = np.unique(field_arr).astype(int).tolist()

        return spw_list, field_list, pol_list

    def read_ms(
        self,
        filepath,
        data_column="DATA",
        pol_order="AIPS",
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        ignore_single_chan=True,
        raise_error=True,
        read_weights=True,
    ):
        """
        Read in a casa measurement set.

        Parameters
        ----------
        filepath : str
            The measurement set root directory to read from.
        data_column : str
            name of CASA data column to read into data_array. Options are:
            'DATA', 'MODEL', or 'CORRECTED_DATA'
        pol_order : str
            Option to specify polarizations order convention, options are
            'CASA' or 'AIPS'.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        ignore_single_chan : bool
            Some measurement sets (e.g., those from ALMA) use single channel spectral
            windows for recording pseudo-continuum channels or storing other metadata
            in the track when the telescopes are not on source. Because of the way
            the object is strutured (where all spectral windows are assumed to be
            simultaneously recorded), this can significantly inflate the size and memory
            footprint of UVData objects. By default, single channel windows are ignored
            to avoid this issue, although they can be included if setting this parameter
            equal to True.
        raise_error : bool
            The measurement set format allows for different spectral windows and
            polarizations to have different metdata for the same time-baseline
            combination, but UVData objects do not. If detected, by default the reader
            will throw an error. However, if set to False, the reader will simply give
            a warning, and will use the first value read in the file as the "correct"
            metadata in the UVData object.
        read_weights : bool
            Read in the weights from the MS file, default is True. If false, the method
            will set the `nsamples_array` to the same uniform value (namely 1.0).

        Raises
        ------
        IOError
            If root file directory doesn't exist.
        ValueError
            If the `data_column` is not set to an allowed value.
            If the data are have multiple subarrays or are multi source or have
            multiple spectral windows.
            If the data have multiple data description ID values.

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        if not os.path.exists(filepath):
            raise IOError(filepath + " not found")
        # set filename variable
        basename = filepath.rstrip("/")
        self.filename = [os.path.basename(basename)]
        self._filename.form = (1,)

        # get the history info
        pyuvdata_written = False
        if tables.tableexists(filepath + "/HISTORY"):
            self.history, pyuvdata_written = self._ms_hist_to_string(
                tables.table(filepath + "/HISTORY", ack=False)
            )
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str
        else:
            self.history = self.pyuvdata_version_str

        data_desc_dict = {}
        tb_desc = tables.table(filepath + "/DATA_DESCRIPTION", ack=False)
        for idx in range(tb_desc.nrows()):
            data_desc_dict[idx] = {
                "SPECTRAL_WINDOW_ID": tb_desc.getcell("SPECTRAL_WINDOW_ID", idx),
                "POLARIZATION_ID": tb_desc.getcell("POLARIZATION_ID", idx),
                "FLAG_ROW": tb_desc.getcell("FLAG_ROW", idx),
            }
        tb_desc.close()

        # Polarization array
        tb_pol = tables.table(filepath + "/POLARIZATION", ack=False)
        for key in data_desc_dict.keys():
            pol_id = data_desc_dict[key]["POLARIZATION_ID"]
            data_desc_dict[key]["CORR_TYPE"] = tb_pol.getcell(
                "CORR_TYPE", pol_id
            ).astype(int)
            data_desc_dict[key]["NUM_CORR"] = tb_pol.getcell("NUM_CORR", pol_id)
        tb_pol.close()

        tb_spws = tables.table(filepath + "/SPECTRAL_WINDOW", ack=False)

        try:
            spw_id = tb_spws.getcol("ASSOC_SPW_ID")
            use_assoc_id = True
        except RuntimeError:
            use_assoc_id = False

        single_chan_list = []
        for key in data_desc_dict.keys():
            spw_id = data_desc_dict[key]["SPECTRAL_WINDOW_ID"]
            data_desc_dict[key]["CHAN_FREQ"] = tb_spws.getcell("CHAN_FREQ", spw_id)
            # beware! There are possibly 3 columns here that might be the correct one
            # to use: CHAN_WIDTH, EFFECTIVE_BW, RESOLUTION
            data_desc_dict[key]["CHAN_WIDTH"] = tb_spws.getcell("CHAN_WIDTH", spw_id)
            data_desc_dict[key]["NUM_CHAN"] = tb_spws.getcell("NUM_CHAN", spw_id)
            if data_desc_dict[key]["NUM_CHAN"] == 1:
                single_chan_list.append(key)
            if use_assoc_id:
                data_desc_dict[key]["SPW_ID"] = int(
                    tb_spws.getcell("ASSOC_SPW_ID", spw_id)[0]
                )
            else:
                data_desc_dict[key]["SPW_ID"] = spw_id

        if ignore_single_chan:
            for key in single_chan_list:
                del data_desc_dict[key]

        # FITS uvw direction convention is opposite ours and Miriad's.
        # CASA's convention is unclear: the docs contradict themselves,
        # but after a question to the helpdesk we got a clear response that
        # the convention is antenna2_pos - antenna1_pos, so the convention is the
        # same as ours & Miriad's
        # HOWEVER, it appears that CASA's importuvfits task does not make this
        # convention change. So if the data in the MS came via that task and was not
        # written by pyuvdata, we do need to flip the uvws & conjugate the data
        flip_conj = ("importuvfits" in self.history) and (not pyuvdata_written)
        spw_list, field_list, pol_list = self._read_ms_main(
            filepath,
            data_column,
            data_desc_dict,
            read_weights=read_weights,
            flip_conj=flip_conj,
            raise_error=raise_error,
        )

        if (spw_list is None) and (field_list is None) and (pol_list is None):
            raise ValueError(
                "No valid data available in the MS file. If this file contains "
                "single channel data, set ignore_single_channel=True when calling "
                "read_ms."
            )

        self.Npols = len(pol_list)
        self.polarization_array = np.array(pol_list, dtype=np.int64)
        self.Nspws = len(spw_list)
        self.spw_array = np.array(spw_list, dtype=np.int64)

        self.Nfreqs = len(self.flex_spw_id_array)
        self.freq_array = np.zeros(self.Nfreqs)
        self.channel_width = np.zeros(self.Nfreqs)

        for key in data_desc_dict.keys():
            sel_mask = self.flex_spw_id_array == data_desc_dict[key]["SPW_ID"]
            self.freq_array[sel_mask] = data_desc_dict[key]["CHAN_FREQ"]
            self.channel_width[sel_mask] = data_desc_dict[key]["CHAN_WIDTH"]

        if (np.unique(self.channel_width).size == 1) and (len(spw_list) == 1):
            # If we don't _need_ the flex_spw setup, then the default is not
            # to use it.
            self.channel_width = float(self.channel_width[0])
            self.flex_spw_id_array = None
        else:
            self._set_flex_spw()

        self.Ntimes = int(np.unique(self.time_array).size)
        self.Nblts = int(self.data_array.shape[0])
        self.Nants_data = len(
            np.unique(
                np.concatenate(
                    (np.unique(self.ant_1_array), np.unique(self.ant_2_array))
                )
            )
        )
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )
        self.Nbls = len(np.unique(self.baseline_array))

        # open table with antenna location information
        tb_ant = tables.table(filepath + "/ANTENNA", ack=False)
        tb_obs = tables.table(filepath + "/OBSERVATION", ack=False)
        self.telescope_name = tb_obs.getcol("TELESCOPE_NAME")[0]
        self.instrument = tb_obs.getcol("TELESCOPE_NAME")[0]
        self.extra_keywords["observer"] = tb_obs.getcol("OBSERVER")[0]
        full_antenna_positions = tb_ant.getcol("POSITION")
        n_ants_table = full_antenna_positions.shape[0]
        xyz_telescope_frame = tb_ant.getcolkeyword("POSITION", "MEASINFO")["Ref"]

        # Note: measurement sets use the antenna number as an index into the antenna
        # table. This means that if the antenna numbers do not start from 0 and/or are
        # not contiguous, empty rows are inserted into the antenna table
        # these 'dummy' rows have positions of zero and need to be removed.
        # (this is somewhat similar to miriad)
        ant_good_position = np.nonzero(np.linalg.norm(full_antenna_positions, axis=1))[
            0
        ]
        full_antenna_positions = full_antenna_positions[ant_good_position, :]
        self.antenna_numbers = np.arange(n_ants_table)[ant_good_position]

        # check to see if a TELESCOPE_LOCATION column is present in the observation
        # table. This is non-standard, but inserted by pyuvdata
        if "TELESCOPE_LOCATION" in tb_obs.colnames():
            self.telescope_location = np.squeeze(tb_obs.getcol("TELESCOPE_LOCATION"))
        else:
            if self.telescope_name in self.known_telescopes():
                # Try to get it from the known telescopes
                self.set_telescope_params()
            elif xyz_telescope_frame == "ITRF":
                # Set it to be the mean of the antenna positions (this is not ideal!)
                self.telescope_location = np.array(
                    np.mean(full_antenna_positions, axis=0)
                )
            else:
                raise ValueError(
                    "Telescope frame is not ITRF and telescope is not "
                    "in known_telescopes, so telescope_location is not set."
                )
        tb_obs.close()

        # antenna names
        ant_names = np.asarray(tb_ant.getcol("NAME"))[ant_good_position].tolist()
        station_names = np.asarray(tb_ant.getcol("STATION"))[ant_good_position].tolist()
        self.antenna_diameters = tb_ant.getcol("DISH_DIAMETER")[ant_good_position]

        # importuvfits measurement sets store antenna names in the STATION column.
        # cotter measurement sets store antenna names in the NAME column, which is
        # inline with the MS definition doc. In that case all the station names are
        # the same. Default to using what the MS definition doc specifies, unless
        # we read importuvfits in the history, or if the antenna column is not filled.
        if ("importuvfits" not in self.history) and (
            len(ant_names) == len(np.unique(ant_names)) and ("" not in ant_names)
        ):
            self.antenna_names = ant_names
        else:
            self.antenna_names = station_names

        self.Nants_telescope = len(self.antenna_names)

        relative_positions = np.zeros_like(full_antenna_positions)
        relative_positions = full_antenna_positions - self.telescope_location.reshape(
            1, 3
        )
        self.antenna_positions = relative_positions

        tb_ant.close()

        self._set_phased()

        # set LST array from times and itrf
        proc = self.set_lsts_from_time_array(background=background_lsts)

        # CASA weights column keeps track of number of data points averaged.

        tb_field = tables.table(filepath + "/FIELD", ack=False)

        # Error if the phase_dir has a polynomial term because we don't know
        # how to handle that
        message = (
            "PHASE_DIR is expressed as a polynomial. "
            "We do not currently support this mode, please make an issue."
        )
        assert tb_field.getcol("PHASE_DIR").shape[1] == 1, message

        # MSv2.0 appears to assume J2000. Not sure how to specifiy otherwise
        measinfo_keyword = tb_field.getcolkeyword("PHASE_DIR", "MEASINFO")

        ref_dir_colname = None
        ref_dir_dict = None
        if "VarRefCol" in measinfo_keyword.keys():
            # This seems to be a yet-undocumented feature for CASA, which allows one
            # to specify an additional (optional?) column that defines the reference
            # frame on a per-source basis.
            ref_dir_colname = measinfo_keyword["VarRefCol"]
            ref_dir_dict = {
                key: name
                for key, name in zip(
                    measinfo_keyword["TabRefCodes"], measinfo_keyword["TabRefTypes"]
                )
            }

        if "Ref" in measinfo_keyword.keys():
            frame_tuple = self._parse_casa_frame_ref(measinfo_keyword["Ref"])
            self.phase_center_frame = frame_tuple[0]
            self.phase_center_epoch = frame_tuple[1]
        else:
            warnings.warn("Coordinate reference frame not detected, defaulting to ICRS")
            self.phase_center_frame = "icrs"
            self.phase_center_epoch = 2000.0

        # If only dealing with a single target, assume we don't want to make a
        # multi-phase-center data set.
        if (len(field_list) == 1) and (ref_dir_dict is None):
            radec_center = tb_field.getcell("PHASE_DIR", field_list[0])[0]
            self.phase_center_ra = float(radec_center[0])
            self.phase_center_dec = float(radec_center[1])
            self.phase_center_id_array = None
            self.object_name = tb_field.getcol("NAME")[0]
        else:
            self._set_multi_phase_center()
            field_id_dict = {field_idx: field_idx for field_idx in field_list}
            try:
                id_arr = tb_field.getcol("SOURCE_ID")
                if np.all(id_arr >= 0) and len(np.unique(id_arr)) == len(id_arr):
                    for idx, sou_id in enumerate(id_arr):
                        if idx in field_list:
                            field_id_dict[idx] = sou_id
            except RuntimeError:
                # Reach here if no column named SOURCE_ID exists, or if it does exist
                # but is completely unfilled. Nothing to do at this point but move on.
                pass
            # Field names are allowed to be the same in CASA, so if we detect
            # conflicting names here we use the FIELD row numbers to try and
            # differentiate between them.
            field_name_list = tb_field.getcol("NAME")
            uniq_names, uniq_count = np.unique(field_name_list, return_counts=True)
            rep_name_list = uniq_names[uniq_count > 1]
            for rep_name in rep_name_list:
                rep_count = 0
                for idx in range(len(field_name_list)):
                    if field_name_list[idx] == rep_name:
                        field_name_list[idx] = "%s-%03i" % (
                            field_name_list[idx],
                            rep_count,
                        )
                        rep_count += 1

            for field_idx in field_list:
                radec_center = tb_field.getcell("PHASE_DIR", field_idx)[0]
                field_name = field_name_list[field_idx]
                if ref_dir_colname is not None:
                    frame_tuple = self._parse_casa_frame_ref(
                        ref_dir_dict[tb_field.getcell(ref_dir_colname, field_list[0])]
                    )
                else:
                    frame_tuple = (self.phase_center_frame, self.phase_center_epoch)

                self._add_phase_center(
                    field_name,
                    cat_type="sidereal",
                    cat_lon=radec_center[0],
                    cat_lat=radec_center[1],
                    cat_frame=frame_tuple[0],
                    cat_epoch=frame_tuple[1],
                    info_source="file",
                    cat_id=field_id_dict[field_idx],
                )
            # Only thing left to do is to update the IDs for the per-BLT records
            self.phase_center_id_array = np.array(
                [field_id_dict[sou_id] for sou_id in self.phase_center_id_array],
                dtype=int,
            )

        tb_field.close()

        if proc is not None:
            proc.join()
        # Fill in the apparent coordinates here
        self._set_app_coords_helper()

        self.data_array = np.expand_dims(self.data_array, 1)
        self.nsample_array = np.expand_dims(self.nsample_array, 1)
        self.flag_array = np.expand_dims(self.flag_array, 1)
        self.freq_array = np.expand_dims(self.freq_array, 0)

        # order polarizations
        self.reorder_pols(order=pol_order, run_check=False)

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
            )
