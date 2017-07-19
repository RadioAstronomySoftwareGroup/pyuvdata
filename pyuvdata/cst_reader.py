from uvbeam import UVBeam
import numpy as np, healpy as hp
import os, sys

def healpixellize(f_in, theta_in, phi_in, nside):
    f = f_in.flatten()
    theta = theta_in.flatten()
    phi = phi_in.flatten()

    pix = hp.ang2pix(nside,theta,phi)

    hmap = np.zeros(hp.nside2npix(nside))
    hits = np.zeros(hp.nside2npix(nside))


    for i,v in enumerate(f):
        hmap[pix[i]] += v
        hits[pix[i]] +=1
    hmap = hmap/hits

    return hmap

def AzimuthalRotation(hmap):
    """
    Azimuthal rotation of a healpix map by pi/2 about the z-axis
    """
    npix = len(hmap)
    nside= hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    t2,p2 = hp.pix2ang(nside, hpxidx)

    p = p2 - np.pi/2
    p[p < 0] += 2. * np.pi
    t = t2

    idx = hp.ang2pix(nside, t, p)

    hout = hmap[idx]
    return hout

class CSTPowerReader(UVBeam):
    """
    A class to extract power beams for two feeds from CST text files.
    Assumes the structure in the simulation was symmetric under
    45 degree rotations about the z-axis.
    """

    def read_cst_files(self, filelist, data_normalization):
        if data_normalization not in ['peak', 'solid_angle']:
            raise ValueError('data_normalization must be specified as either "peak" or "solid_angle"')

        self.Naxes_vec = 1
        self.Nfreqs = len(filelist)
        self.Nspws = 1
        self.antenna_type = 'simple'
        self.beam_type = 'power'
        self.data_normalization = data_normalization
        self.feed_name = 'bob'
        self.feed_version = 'unknown'

        self.freq_array = []

        self.history = 'In the beginning...'
        self.model_name = 'E-field pattern - Rigging height 4.9m'
        self.model_version = '1.0'
        self.pixel_coordinate_system = 'healpix'

        self.spw_array = np.array([1])

        self.telescope_name = 'HERA'
        self.nside = 32
        self.Npixels = 12 * 32**2
        self.Npols = 2
        self.ordering = 'ring'
        self.pixel_array = np.arange(self.Npixels)
        self.polarization_array = [-5,-6]
        self.pixel_coordinate_system = 'healpix'

        self.data_array = np.zeros((self.Naxes_vec,
                                    self.Nspws,
                                    self.Npols,
                                    self.Nfreqs,
                                    self.Npixels),dtype=np.complex)

        for i,fname in enumerate(filelist):
            self.freq_array.append(self.name2freq(fname))
            beam = self.read_cst_power2healpix(fname)
            beam2 = AzimuthalRotation(beam)

            if self.data_normalization == 'peak':
                beam /= np.amax(beam)
                beam2 /= np.amax(beam2)
            elif self.data_normalization == 'solid_angle':
                norm = (4. * np.pi / self.Npixels) * np.sum(beam)
                beam /= norm
                beam2 /= norm

            self.data_array[0,0,0,i,:] = beam
            self.data_array[0,0,1,i,:] = beam2

        self.freq_array = np.array(self.freq_array)
        sort_inds = np.argsort(self.freq_array)

        self.data_array[0,0,0,:,:] = self.data_array[0,0,0,sort_inds,:]
        self.data_array[0,0,1,:,:] = self.data_array[0,0,1,sort_inds,:]

        self.freq_array.sort()
        self.freq_array *= 1e6
        self.freq_array = np.broadcast_to(self.freq_array, (self.Nspws,self.Nfreqs))

    def read_cst_power2healpix(self, filename):
        data = np.loadtxt(filename, skiprows=2)
        theta_data = np.radians(data[:,0])
        phi_data = np.radians(data[:,1])

        power_data = data[:,2]**2.

        theta_f,phi_f = np.abs(theta_data), np.where(theta_data < 0, phi_data + np.pi, phi_data)

        nside = 32
        npix = hp.nside2npix(nside)
        hpxidx = np.arange(npix)
        theta, phi = hp.pix2ang(nside, hpxidx)
        phi = np.where(phi >= np.pi, phi - np.amax(phi), phi)

        power = healpixellize(power_data, theta_f, phi_f, nside)

        return power

    def name2freq(self, fname):
        """
        assumes the file name contains a substring with the frequency
        channel in MHz that the data represents
        e.g. "HERA_Sim_120MHz.txt"
        """
        fi = fname.find('MHz')
        return float(fname[fi-3:fi])
