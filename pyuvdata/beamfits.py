"""Class for reading and writing beamfits files."""
import numpy as np
from astropy.io import fits
from uvbeam import UVBeam


class BeamFITS(UVBeam):
    """
    Defines a beamfits-specific subclass of UVBeam for reading and writing beamfits files.
    This class should not be interacted with directly, instead use the read_beamfits
    and write_beamfits methods on the UVBeam class.

    """

    def write_beamfits(self, filename, run_check=True,
                       run_check_acceptability=True, clobber=False):
        """
        Write the data to a calfits file.

        Args:
            filename: The calfits file to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptability of the values of
                required parameters before writing the file. Default is True.

        """
        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

        if self.Nfreqs > 1:
            freq_spacing = self.freq_array[0, 1:] - self.freq_array[0, :-1]
            if not np.isclose(np.min(freq_spacing), np.max(freq_spacing),
                              rtol=self._freq_array.tols[0], atol=self._freq_array.tols[1]):
                raise ValueError('The frequencies are not evenly spaced (probably '
                                 'because of a select operation). The beamfits format '
                                 'does not support unevenly spaced frequencies.')
        else:
            freq_spacing = 1

        primary_header = fits.Header()
        # Conforming to fits format
        primary_header['SIMPLE'] = True
        primary_header['BITPIX'] = 32

        # metadata
        primary_header['TELESCOP'] = self.telescope_name
        primary_header['FEED'] = self.feed_name
        primary_header['FEEDVER'] = self.feed_version
        primary_header['MODEL'] = self.model_name
        primary_header['MODELVER'] = self.model_version

        # shapes
        primary_header['NFREQS'] = self.Nfreqs
        primary_header['NFEEDS'] = self.Nfeeds
        primary_header['NAXES'] = self.Naxes
        primary_header['NSPWS'] = self.Nspws

        primary_header['COORDSYS'] = self.coordinate_system

        # feed_array: Nfeeds
        # freq_array: Nfreq
        # spw_array: Nspws
        #
        # pixel_location_array: (Naxes, Npixels)
        # basis_vector_array: (Naxes, Npixels)
        # efield_array: (Nfeeds, Naxes, Npixels, Nfreq)

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.iteritems():
            # header keywords have to be 8 characters or less
            keyword = key[:8].upper()
            # print "keyword=-value-", keyword+'='+'-'+str(value)+'-'
            if keyword == 'COMMENT':
                for line in value.splitlines():
                    primary_header.add_comment(line)
            else:
                primary_header[keyword] = value

        for line in self.history.splitlines():
            primary_header.add_history(line)
