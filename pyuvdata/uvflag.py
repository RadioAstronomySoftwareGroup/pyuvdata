# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import print_function, division, absolute_import
import numpy as np
import os
from . import UVData
from . import UVCal
from . import utils as uvutils
from . import version as uvversion
from . import telescopes as uvtel
import warnings
import copy
from six.moves import map


class UVFlag(object):
    ''' Object to handle flag arrays and waterfalls. Supports reading/writing,
    and stores all relevant information to combine flags and apply to data.
    '''

    def __init__(self, input, mode='metric', copy_flags=False, waterfall=False, history='',
                 label=''):
        '''Initialize UVFlag object.
        Args:
            input: UVData object, UVCal object, or path to previously saved UVFlag object.
                   Can also be a list of any compatible combination of the above options.
                   UVData and UVCal objects cannot be directly combined, unless waterfall is True.
            mode: "metric" (default) or "flag" to initialize UVFlag in given mode.
                  The mode determines whether the object has a floating point metric_array
                  or a boolean flag_array.
            copy_flags: Whether to copy flags from input to new UVFlag object. Default is False.
            waterfall: Whether to immediately initialize as a waterfall object,
                       with flag/metric axes: time, frequency, polarization. Default is False.
            history: History string to attach to object.
            label: string used for labeling the object (e.g. 'FM')
        '''

        self.mode = mode.lower()  # Gets overwritten if reading file
        self.history = ''  # Added to at the end
        # String to add to history of any files written with this version of pyuvdata
        self.pyuvdata_version_str = uvversion.version + '.'
        if uvversion.git_hash is not '':
            self.pyuvdata_version_str += ('  Git origin: ' + uvversion.git_origin
                                          + '.  Git hash: ' + uvversion.git_hash
                                          + '.  Git branch: ' + uvversion.git_branch
                                          + '.  Git description: ' + uvversion.git_description + '.')
        self.label = ''  # Added to at the end
        if isinstance(input, (list, tuple)):
            self.__init__(input[0], mode=mode, copy_flags=copy_flags, waterfall=waterfall, history=history)
            if len(input) > 1:
                for i in input[1:]:
                    fobj = UVFlag(i, mode=mode, copy_flags=copy_flags, waterfall=waterfall, history=history)
                    self += fobj
                del(fobj)

        elif isinstance(input, str):
            # Given a path, read input
            self.read(input, history)
        elif waterfall and issubclass(input.__class__, (UVData, UVCal)):
            self.type = 'waterfall'
            self.history += 'Flag object with type "waterfall" created by ' + self.pyuvdata_version_str
            self.time_array, ri = np.unique(input.time_array, return_index=True)
            self.freq_array = input.freq_array[0, :]
            if issubclass(input.__class__, UVData):
                self.polarization_array = input.polarization_array
                self.lst_array = input.lst_array[ri]
            else:
                self.polarization_array = input.jones_array
                self.lst_array = lst_from_uv(input)[ri]
            if copy_flags:
                self.metric_array = flags2waterfall(input, keep_pol=True)
                self.history += ' Waterfall generated from ' + str(input.__class__) + ' object.'
                if self.mode == 'flag':
                    warnings.warn('Copying flags into waterfall results in mode=="metric".')
                    self.mode = 'metric'
            else:
                if self.mode == 'flag':
                    self.flag_array = np.zeros((len(self.time_array),
                                                len(self.freq_array),
                                                len(self.polarization_array)), np.bool)
                elif self.mode == 'metric':
                    self.metric_array = np.zeros((len(self.time_array),
                                                  len(self.freq_array),
                                                  len(self.polarization_array)))

        elif issubclass(input.__class__, UVData):
            self.type = 'baseline'
            self.history += 'Flag object with type "baseline" created by ' + self.pyuvdata_version_str
            self.baseline_array = input.baseline_array
            self.ant_1_array = input.ant_1_array
            self.ant_2_array = input.ant_2_array
            self.time_array = input.time_array
            self.lst_array = input.lst_array
            self.freq_array = input.freq_array
            self.polarization_array = input.polarization_array
            self.Nants_telescope = input.Nants_telescope
            if copy_flags:
                self.flag_array = input.flag_array
                self.history += ' Flags copied from ' + str(input.__class__) + ' object.'
                if self.mode == 'metric':
                    warnings.warn('Copying flags to type=="baseline" results in mode=="flag".')
                    self.mode = 'flag'
            else:
                if self.mode == 'flag':
                    self.flag_array = np.zeros_like(input.flag_array)
                elif self.mode == 'metric':
                    self.metric_array = np.zeros_like(input.flag_array).astype(np.float)

        elif issubclass(input.__class__, UVCal):
            self.type = 'antenna'
            self.history += 'Flag object with type "antenna" created by ' + self.pyuvdata_version_str
            self.ant_array = input.ant_array
            self.time_array = input.time_array
            self.lst_array = lst_from_uv(input)
            self.freq_array = input.freq_array
            self.polarization_array = input.jones_array
            if copy_flags:
                self.flag_array = input.flag_array
                self.history += ' Flags copied from ' + str(input.__class__) + ' object.'
                if self.mode == 'metric':
                    warnings.warn('Copying flags to type=="antenna" results in mode=="flag".')
                    self.mode = 'flag'
            else:
                if self.mode == 'flag':
                    self.flag_array = np.zeros_like(input.flag_array)
                elif self.mode == 'metric':
                    self.metric_array = np.zeros_like(input.flag_array).astype(np.float)

        if not isinstance(input, str):
            if self.mode == 'flag':
                self.weights_array = np.ones(self.flag_array.shape)
            else:
                self.weights_array = np.ones(self.metric_array.shape)

        if history not in self.history:
            self.history += history
        self.label += label

        self.clear_unused_attributes()

    def __eq__(self, other, check_history=False):
        """ Function to check equality of two UVFlag objects
        Args:
            other: UVFlag object to check against
        """
        if not isinstance(other, self.__class__):
            return False
        if (self.type != other.type) or (self.mode != other.mode) or (self.label != other.label):
            return False

        array_list = ['weights_array', 'time_array', 'lst_array', 'freq_array',
                      'polarization_array']
        if self.type == 'antenna':
            array_list += ['ant_array']
        elif self.type == 'baseline':
            array_list += ['baseline_array', 'ant_1_array', 'ant_2_array', 'Nants_telescope']
        if self.mode == 'flag':
            array_list += ['flag_array']
        elif self.mode == 'metric':
            array_list += ['metric_array']
        for arr in array_list:
            self_param = getattr(self, arr)
            other_param = getattr(other, arr)
            if not np.all(self_param == other_param):
                return False

        if check_history:
            if self.history != other.history:
                return False

        return True

    def read(self, filename, history=''):
        """
        Read in flag/metric data from a HDF5 file.

        Args:
            filename: The file name to read.
            history: History string to append to UVFlag history attribute.
        """
        import h5py

        if isinstance(filename, (tuple, list)):
            self.read(filename[0])
            if len(filename) > 1:
                for f in filename[1:]:
                    f2 = UVFlag(f, history=history)
                    self += f2
                del(f2)

        else:
            if not os.path.exists(filename):
                raise IOError(filename + ' not found.')

            # Open file for reading
            with h5py.File(filename, 'r') as f:
                header = f['/Header']

                self.type = uvutils._bytes_to_str(header['type'][()])
                self.mode = uvutils._bytes_to_str(header['mode'][()])
                self.time_array = header['time_array'][()]
                self.lst_array = header['lst_array'][()]
                self.freq_array = header['freq_array'][()]
                self.history = uvutils._bytes_to_str(header['history'][()]) + ' Read by ' + self.pyuvdata_version_str
                self.history += history
                if 'label' in header.keys():
                    self.label = uvutils._bytes_to_str(header['label'][()])
                self.polarization_array = header['polarization_array'][()]
                if self.type == 'baseline':
                    self.baseline_array = header['baseline_array'][()]
                    self.ant_1_array = header['ant_1_array'][()]
                    self.ant_2_array = header['ant_2_array'][()]
                    try:
                        self.Nants_telescope = header['Nants_telescope'][()]
                    except KeyError:
                        warnings.warn('Nants_telescope not available in file, '
                                      'assuming < 2048.')
                        self.Nants_telescope = None
                elif self.type == 'antenna':
                    self.ant_array = header['ant_array'][()]

                dgrp = f['/Data']
                if self.mode == 'metric':
                    self.metric_array = dgrp['metric_array'][()]
                elif self.mode == 'flag':
                    self.flag_array = dgrp['flag_array'][()]

                self.weights_array = dgrp['weights_array'][()]

            self.clear_unused_attributes()

    def write(self, filename, clobber=False, data_compression='lzf'):
        """
        Write a UVFlag object to a hdf5 file.

        Args:
            filename: The file to write to.
            clobber: Option to overwrite the file if it already exists. Default is False.
            data_compression: HDF5 filter to apply when writing the data_array. Default is
                 LZF. If no compression is wanted, set to None.
        """
        import h5py

        if os.path.exists(filename):
            if clobber:
                print('File ' + filename + ' exists; clobbering')
            else:
                raise ValueError('File ' + filename + ' exists; skipping')

        with h5py.File(filename, 'w') as f:
            header = f.create_group('Header')

            # write out metadata
            header['type'] = uvutils._str_to_bytes(self.type)
            header['mode'] = uvutils._str_to_bytes(self.mode)
            header['time_array'] = self.time_array
            header['lst_array'] = self.lst_array
            header['freq_array'] = self.freq_array
            header['polarization_array'] = self.polarization_array
            header['history'] = uvutils._str_to_bytes(self.history + 'Written by ' + self.pyuvdata_version_str)
            header['label'] = uvutils._str_to_bytes(self.label)

            if self.type == 'baseline':
                header['baseline_array'] = self.baseline_array
                header['ant_1_array'] = self.ant_1_array
                header['ant_2_array'] = self.ant_2_array
                header['Nants_telescope'] = self.Nants_telescope
            elif self.type == 'antenna':
                header['ant_array'] = self.ant_array

            dgrp = f.create_group("Data")
            wtsdata = dgrp.create_dataset('weights_array', chunks=True,
                                          data=self.weights_array,
                                          compression=data_compression)
            if self.mode == 'metric':
                data = dgrp.create_dataset('metric_array', chunks=True,
                                           data=self.metric_array,
                                           compression=data_compression)
            elif self.mode == 'flag':
                data = dgrp.create_dataset('flag_array', chunks=True,
                                           data=self.flag_array,
                                           compression=data_compression)

    def __add__(self, other, inplace=False, axis='time'):
        '''Add two UVFlag objects together along a given axis.
        Args:
            other: second UVFlag object to concatenate with self.
            inplace: Whether to concatenate to self, or create a new UVFlag object. Default is False.
            axis: Axis along which to combine UVFlag objects. Default is time.
        '''

        # Handle in place
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that objects are compatible
        if not isinstance(other, this.__class__):
            raise ValueError('Only UVFlag objects can be added to a UVFlag object')
        if this.type != other.type:
            raise ValueError('UVFlag object of type ' + other.type + ' cannot be '
                             'added to object of type ' + this.type + '.')
        if this.mode != other.mode:
            raise ValueError('UVFlag object of mode ' + other.mode + ' cannot be '
                             'added to object of mode ' + this.type + '.')

        # Simplify axis referencing
        axis = axis.lower()
        type_nums = {'waterfall': 0, 'baseline': 1, 'antenna': 2}
        axis_nums = {'time': [0, 0, 3], 'baseline': [None, 0, None],
                     'antenna': [None, None, 0], 'frequency': [1, 2, 2],
                     'polarization': [2, 3, 4], 'pol': [2, 3, 4],
                     'jones': [2, 3, 4]}
        ax = axis_nums[axis][type_nums[self.type]]
        if axis == 'time':
            this.time_array = np.concatenate([this.time_array, other.time_array])
            this.lst_array = np.concatenate([this.lst_array, other.lst_array])
            if this.type == 'baseline':
                this.baseline_array = np.concatenate([this.baseline_array, other.baseline_array])
                this.ant_1_array = np.concatenate([this.ant_1_array, other.ant_1_array])
                this.ant_2_array = np.concatenate([this.ant_2_array, other.ant_2_array])
        elif axis == 'baseline':
            if self.type != 'baseline':
                raise ValueError('Flag object of type ' + self.type + ' cannot be '
                                 'concatenated along baseline axis.')
            this.time_array = np.concatenate([this.time_array, other.time_array])
            this.lst_array = np.concatenate([this.lst_array, other.lst_array])
            this.baseline_array = np.concatenate([this.baseline_array, other.baseline_array])
            this.ant_1_array = np.concatenate([this.ant_1_array, other.ant_1_array])
            this.ant_2_array = np.concatenate([this.ant_2_array, other.ant_2_array])
        elif axis == 'antenna':
            if self.type != 'antenna':
                raise ValueError('Flag object of type ' + self.type + ' cannot be '
                                 'concatenated along antenna axis.')
            this.ant_array = np.concatenate([this.ant_array, other.ant_array])
        elif axis == 'frequency':
            this.freq_array = np.concatenate([this.freq_array, other.freq_array])
        elif axis in ['polarization', 'pol', 'jones']:
            this.polarization_array = np.concatenate([this.polarization_array,
                                                      other.polarization_array])

        if this.mode == 'flag':
            this.flag_array = np.concatenate([this.flag_array, other.flag_array],
                                             axis=ax)
        elif this.mode == 'metric':
            this.metric_array = np.concatenate([this.metric_array,
                                                other.metric_array], axis=ax)
        this.weights_array = np.concatenate([this.weights_array,
                                             other.weights_array], axis=ax)
        this.history += 'Data combined along ' + axis + ' axis with ' + self.pyuvdata_version_str

        if not inplace:
            return this

    def __iadd__(self, other, axis='time'):
        """
        In place add.

        Args:
            other: Another UVFlag object which will be added to self.
            axis: Axis along which to combine UVFlag objects. Default is time.
        """
        self.__add__(other, inplace=True, axis=axis)
        return self

    def __or__(self, other, inplace=False):
        '''Combine two UVFlag objects in "flag" mode by "OR"-ing their flags.
        Args:
            other: second UVFlag object to combine with self.
            inplace: Whether to combine to self, or create a new UVFlag object. Default is False.
        '''
        if (self.mode != 'flag') or (other.mode != 'flag'):
            raise ValueError('UVFlag object must be in "flag" mode to use "or" function.')

        # Handle in place
        if inplace:
            this = self
        else:
            this = self.copy()
        this.flag_array += other.flag_array
        if other.history not in this.history:
            this.history += "Flags OR'd with: " + other.history

        if not inplace:
            return this

    def __ior__(self, other):
        '''In place or
        Args:
            other: second UVFlag object to combine with self.
        '''
        self.__or__(other, inplace=True)
        return self

    def clear_unused_attributes(self):
        """
        Remove unused attributes. Useful when changing type or mode.
        """
        if hasattr(self, 'baseline_array') and self.type != 'baseline':
            del(self.baseline_array)
        if hasattr(self, 'ant_1_array') and self.type != 'baseline':
            del(self.ant_1_array)
        if hasattr(self, 'ant_2_array') and self.type != 'baseline':
            del(self.ant_2_array)
        if hasattr(self, 'Nants_telescope') and self.type != 'baseline':
            del(self.Nants_telescope)
        if hasattr(self, 'ant_array') and self.type != 'antenna':
            del(self.ant_array)
        if hasattr(self, 'metric_array') and self.mode != 'metric':
            del(self.metric_array)
        if hasattr(self, 'flag_array') and self.mode != 'flag':
            del(self.flag_array)

    def copy(self):
        ''' Simply return a copy of this object '''
        return copy.deepcopy(self)

    def combine_metrics(self, others, method='quadmean', inplace=True):
        """
        Combine metric arrays between different UVFlag objects together.
        Args:
            others (UVFlag or list of UVFlags): Other UVFlag objects to combine
                metrics with this one.
            method (str, optional): Method to combine metrics. Default is "quadmean".
            inplace (bool, optional): Perform combination in place. Default is True.
        Returns:
            uvf (UVFlag): If inplace==False, return new UVFlag object with combined metrics.
        """
        # Ensure others is iterable (in case of single UVFlag object)
        others = uvutils._get_iterable(others)
        if np.any([not isinstance(other, UVFlag) for other in others]):
            raise ValueError('"others" must be UVFlag or list of UVFlag objects')
        if (self.mode != 'metric') or np.any([other.mode != 'metric' for other in others]):
            raise ValueError('UVFlag object and "others" must be in "flag" mode '
                             'to use "add_metrics" function.')
        if inplace:
            this = self
        else:
            this = self.copy()
        method = method.lower()
        darray = np.expand_dims(this.metric_array, 0)
        warray = np.expand_dims(this.weights_array, 0)
        for other in others:
            if this.metric_array.shape != other.metric_array.shape:
                raise ValueError('UVFlag metric array shapes do not match.')
            darray = np.vstack([darray, np.expand_dims(other.metric_array, 0)])
            warray = np.vstack([warray, np.expand_dims(other.weights_array, 0)])
        darray, warray = uvutils.collapse(darray, method, weights=warray, axis=0, return_weights=True)
        this.metric_array = darray
        this.weights_array = warray
        this.history += 'Combined metric arrays using ' + self.pyuvdata_version_str
        if not inplace:
            return this

    def collapse_pol(self, method='quadmean'):
        """
        Collapse the polarization axis using a given method.

        If the original UVFlag object has more than one polarization,
        the resulting polarization_array will be a single element array with a
        comma separated string encoding the original polarizations.

        Args:
            method: How to collapse the dimension(s)
        """
        method = method.lower()
        if self.mode == 'flag':
            darr = self.flag_array
        else:
            darr = self.metric_array
        if len(self.polarization_array) > 1:
            # Collapse pol dimension. But note we retain a polarization axis.
            d, w = uvutils.collapse(darr, method, axis=-1, weights=self.weights_array, return_weights=True)
            darr = np.expand_dims(d, axis=d.ndim)
            self.weights_array = np.expand_dims(w, axis=w.ndim)
            self.polarization_array = np.array([','.join(map(str, self.polarization_array))],
                                               dtype=np.string_)
        else:
            warnings.warn('Cannot collapse polarization axis when only one pol present.')
            return
        if ((method == 'or') or (method == 'and')) and (self.mode == 'flag'):
            self.flag_array = darr
        else:
            self.metric_array = darr
            self.mode = 'metric'
        self.clear_unused_attributes()
        self.history += 'Pol axis collapsed with ' + self.pyuvdata_version_str

    def to_waterfall(self, method='quadmean', keep_pol=True):
        """
        Convert an 'antenna' or 'baseline' type object to waterfall using a given method.
        Args:
            method: How to collapse the dimension(s)
            keep_pol: Whether to also collapse the polarization dimension
                      If keep_pol is False, and the original UVFlag object has more
                      than one polarization, the resulting polarization_array
                      will be a single element array with a comma separated string
                      encoding the original polarizations.
        """
        method = method.lower()
        if self.type == 'waterfall' and (keep_pol or (len(self.polarization_array) == 1)):
            warnings.warn('This object is already a waterfall. Nothing to change.')
            return
        if (not keep_pol) and (len(self.polarization_array) > 1):
            self.collapse_pol(method)

        if self.mode == 'flag':
            darr = self.flag_array
        else:
            darr = self.metric_array

        if self.type == 'antenna':
            d, w = uvutils.collapse(darr, method, axis=(0, 1), weights=self.weights_array,
                                    return_weights=True)
            darr = np.swapaxes(d, 0, 1)
            self.weights_array = np.swapaxes(w, 0, 1)
        elif self.type == 'baseline':
            Nt = len(np.unique(self.time_array))
            Nf = len(self.freq_array[0, :])
            Np = len(self.polarization_array)
            d = np.zeros((Nt, Nf, Np))
            w = np.zeros((Nt, Nf, Np))
            for i, t in enumerate(np.unique(self.time_array)):
                ind = self.time_array == t
                d[i, :, :], w[i, :, :] = uvutils.collapse(darr[ind, :, :], method,
                                                          axis=0,
                                                          weights=self.weights_array[ind, :, :],
                                                          return_weights=True)
            darr = d
            self.weights_array = w
            self.time_array, ri = np.unique(self.time_array, return_index=True)
            self.lst_array = self.lst_array[ri]
        if ((method == 'or') or (method == 'and')) and (self.mode == 'flag'):
            # If using a boolean operation (AND/OR) and in flag mode, stay in flag
            self.flag_array = darr
        else:
            # Otherwise change to (or stay in) metric
            self.metric_array = darr
            self.mode = 'metric'
        self.freq_array = self.freq_array.flatten()
        self.type = 'waterfall'
        self.history += 'Collapsed to type "waterfall" with ' + self.pyuvdata_version_str
        self.clear_unused_attributes()

    def to_baseline(self, uv, force_pol=False):
        '''Convert a UVFlag object of type "waterfall" to type "baseline".
        Broadcasts the flag array to all baselines.
        This function does NOT apply flags to uv.
        Args:
            uv: UVData or UVFlag object of type baseline to match.
            force_pol: If True, will use 1 pol to broadcast to any other pol.
                       Otherwise, will require polarizations match.
                       For example, this keyword is useful if one flags on all
                       pols combined, and wants to broadcast back to individual pols.
        '''
        if self.type == 'baseline':
            return
        if not (issubclass(uv.__class__, UVData) or (isinstance(uv, UVFlag) and uv.type == 'baseline')):
            raise ValueError('Must pass in UVData object or UVFlag object of type '
                             '"baseline" to match.')
        if self.type != 'waterfall':
            raise ValueError('Cannot convert from type "' + self.type + '" to "baseline".')
        # Deal with polarization
        if force_pol and self.polarization_array.size == 1:
            # Use single pol for all pols, regardless
            self.polarization_array = uv.polarization_array
            # Broadcast arrays
            if self.mode == 'flag':
                self.flag_array = self.flag_array.repeat(self.polarization_array.size, axis=-1)
            else:
                self.metric_array = self.metric_array.repeat(self.polarization_array.size, axis=-1)
            self.weights_array = self.weights_array.repeat(self.polarization_array.size, axis=-1)
        # Now the pol axes should match regardless of force_pol.
        if not np.array_equal(uv.polarization_array, self.polarization_array):
            if self.polarization_array.size == 1:
                raise ValueError('Polarizations do not match. Try keyword force_pol'
                                 + ' if you wish to broadcast to all polarizations.')
            else:
                raise ValueError('Polarizations could not be made to match.')
        # Populate arrays
        warr = np.zeros_like(uv.flag_array)
        if self.mode == 'flag':
            arr = np.zeros_like(uv.flag_array)
            sarr = self.flag_array
        elif self.mode == 'metric':
            arr = np.zeros_like(uv.flag_array, dtype=float)
            sarr = self.metric_array
        for i, t in enumerate(np.unique(uv.time_array)):
            ti = np.where(uv.time_array == t)
            arr[ti, :, :, :] = sarr[i, :, :][np.newaxis, np.newaxis, :, :]
            warr[ti, :, :, :] = self.weights_array[i, :, :][np.newaxis, np.newaxis, :, :]
        if self.mode == 'flag':
            self.flag_array = arr
        elif self.mode == 'metric':
            self.metric_array = arr
        self.weights_array = warr

        self.baseline_array = uv.baseline_array
        self.ant_1_array = uv.ant_1_array
        self.ant_2_array = uv.ant_2_array
        self.time_array = uv.time_array
        self.lst_array = uv.lst_array
        self.Nants_telescope = uv.Nants_telescope
        self.type = 'baseline'
        self.history += 'Broadcast to type "baseline" with ' + self.pyuvdata_version_str

    def to_antenna(self, uv, force_pol=False):
        '''Convert a UVFlag object of type "waterfall" to type "antenna".
        Broadcasts the flag array to all antennas.
        This function does NOT apply flags to uv.
        Args:
            uv: UVCal or UVFlag object of type antenna to match.
            force_pol: If True, will use 1 pol to broadcast to any other pol.
                       Otherwise, will require polarizations match.
                       For example, this keyword is useful if one flags on all
                       pols combined, and wants to broadcast back to individual pols.
        '''
        if self.type == 'antenna':
            return
        if not (issubclass(uv.__class__, UVCal) or (isinstance(uv, UVFlag) and uv.type == 'antenna')):
            raise ValueError('Must pass in UVCal object or UVFlag object of type '
                             '"antenna" to match.')
        if self.type != 'waterfall':
            raise ValueError('Cannot convert from type "' + self.type + '" to "antenna".')
        # Deal with polarization
        if issubclass(uv.__class__, UVCal):
            polarr = uv.jones_array
        else:
            polarr = uv.polarization_array
        if force_pol and self.polarization_array.size == 1:
            # Use single pol for all pols, regardless
            self.polarization_array = polarr
            # Broadcast arrays
            if self.mode == 'flag':
                self.flag_array = self.flag_array.repeat(self.polarization_array.size, axis=-1)
            else:
                self.metric_array = self.metric_array.repeat(self.polarization_array.size, axis=-1)
            self.weights_array = self.weights_array.repeat(self.polarization_array.size, axis=-1)
        # Now the pol axes should match regardless of force_pol.
        if not np.array_equal(polarr, self.polarization_array):
            if self.polarization_array.size == 1:
                raise ValueError('Polarizations do not match. Try keyword force_pol'
                                 + 'if you wish to broadcast to all polarizations.')
            else:
                raise ValueError('Polarizations could not be made to match.')
        # Populate arrays
        if self.mode == 'flag':
            self.flag_array = np.swapaxes(self.flag_array, 0, 1)[np.newaxis, np.newaxis,
                                                                 :, :, :]
            self.flag_array = self.flag_array.repeat(len(uv.ant_array), axis=0)
        elif self.mode == 'metric':
            self.metric_array = np.swapaxes(self.metric_array, 0, 1)[np.newaxis, np.newaxis,
                                                                     :, :, :]
            self.metric_array = self.metric_array.repeat(len(uv.ant_array), axis=0)
        self.weights_array = np.swapaxes(self.weights_array, 0, 1)[np.newaxis, np.newaxis,
                                                                   :, :, :]
        self.weights_array = self.weights_array.repeat(len(uv.ant_array), axis=0)
        self.ant_array = uv.ant_array
        self.type = 'antenna'
        self.history += 'Broadcast to type "antenna" with ' + self.pyuvdata_version_str

    def to_flag(self, threshold=np.inf):
        '''Convert to flag mode. NOT SMART. Removes metric_array and creates a
        flag_array from a simple threshold on the metric values.

        Args:
            threshold (float): Metric value over which the corresponding flag is
                set to True. Default is np.inf, which results in flags of all False.
        '''
        if self.mode == 'flag':
            return
        elif self.mode == 'metric':
            self.flag_array = np.where(self.metric_array >= threshold,
                                       True, False)
            self.mode = 'flag'
            self.weights_array = np.ones_like(self.metric_array, dtype=np.float)
        else:
            raise ValueError('Unknown UVFlag mode: ' + self.mode + '. Cannot convert to flag.')
        self.history += 'Converted to mode "flag" with ' + self.pyuvdata_version_str
        self.clear_unused_attributes()

    def to_metric(self, convert_wgts=False):
        '''Convert to metric mode. NOT SMART. Simply recasts flag_array as float
        and uses this as the metric array.

        Args:
            convert_wgts : bool, if True convert self.weights_array to ones
                unless a column or row is completely flagged, in which case
                convert those pixels to zero. This is used when reinterpretting
                flags as metrics to calculate flag fraction. Zero weighting
                completely flagged rows/columns prevents those from counting
                against a threshold along the other dimension.
        '''
        if self.mode == 'metric':
            return
        elif self.mode == 'flag':
            self.metric_array = self.flag_array.astype(np.float)
            self.mode = 'metric'
            if convert_wgts:
                self.weights_array = np.ones_like(self.weights_array)
                if self.type == 'waterfall':
                    for i, pol in enumerate(self.polarization_array):
                        self.weights_array[:, :, i] *= ~and_rows_cols(self.flag_array[:, :, i])
                elif self.type == 'baseline':
                    for i, pol in enumerate(self.polarization_array):
                        for j, ap in enumerate(self.get_antpairs()):
                            inds = self.antpair2ind(*ap)
                            self.weights_array[inds, 0, :, i] *= ~and_rows_cols(self.flag_array[inds, 0, :, i])
                elif self.type == 'antenna':
                    for i, pol in enumerate(self.polarization_array):
                        for j in range(self.weights_array.shape[0]):
                            self.weights_array[j, 0, :, :, i] *= ~and_rows_cols(self.flag_array[j, 0, :, :, i])
        else:
            raise ValueError('Unknown UVFlag mode: ' + self.mode + '. Cannot convert to metric.')
        self.history += 'Converted to mode "metric" with ' + self.pyuvdata_version_str
        self.clear_unused_attributes()

    def antpair2ind(self, ant1, ant2):
        """
        Get blt indices for given (ordered) antenna pair.
        """
        if self.type != 'baseline':
            raise ValueError('UVFlag object of type ' + self.type + ' does not '
                             'contain antenna pairs to index.')
        return np.where((self.ant_1_array == ant1) & (self.ant_2_array == ant2))[0]

    def baseline_to_antnums(self, baseline):
        """
        Get the antenna numbers corresponding to a given baseline number.

        Args:
            baseline(int): baseline number

        Returns:
            (tuple): Antenna numbers corresponding to baseline.
        """
        assert self.type == 'baseline', 'Must be "baseline" type UVFlag object.'
        return uvutils.baseline_to_antnums(baseline, self.Nants_telescope)

    def get_baseline_nums(self):
        """
        Returns numpy array of unique baseline numbers in data.
        """
        assert self.type == 'baseline', 'Must be "baseline" type UVFlag object.'
        return np.unique(self.baseline_array)

    def get_antpairs(self):
        """
        Returns list of unique antpair tuples (ant1, ant2) in data.
        """
        assert self.type == 'baseline', 'Must be "baseline" type UVFlag object.'
        return [self.baseline_to_antnums(bl) for bl in self.get_baseline_nums()]


def flags2waterfall(uv, flag_array=None, keep_pol=False):
    """
    Convert a flag array to a 2D waterfall of dimensions (Ntimes, Nfreqs).
    Averages over baselines and polarizations (in the case of visibility data),
    or antennas and jones parameters (in case of calibrationd data).
    Args:
        uv -- A UVData or UVCal object which defines the times and frequencies,
              and supplies the flag_array to convert (if flag_array not specified)
        flag_array -- Optional flag array to convert instead of uv.flag_array.
                      Must have same dimensions as uv.flag_array.
        keep_pol -- Option to keep the polarization axis intact. Default is False.
    Returns:
        waterfall -- 2D waterfall of averaged flags, for example fraction of baselines
                     which are flagged for every time and frequency (in case of UVData input)
                     Size is (Ntimes, Nfreqs) or (Ntimes, Nfreqs, Npols).
    """
    if not isinstance(uv, (UVData, UVCal)):
        raise ValueError('flags2waterfall() requires a UVData or UVCal object as '
                         'the first argument.')
    if flag_array is None:
        flag_array = uv.flag_array
    if uv.flag_array.shape != flag_array.shape:
        raise ValueError('Flag array must align with UVData or UVCal object.')

    if isinstance(uv, UVCal):
        if keep_pol:
            waterfall = np.swapaxes(np.mean(flag_array, axis=(0, 1)), 0, 1)
        else:
            waterfall = np.mean(flag_array, axis=(0, 1, 4)).T
    else:
        if keep_pol:
            waterfall = np.zeros((uv.Ntimes, uv.Nfreqs, uv.Npols))
            for i, t in enumerate(np.unique(uv.time_array)):
                waterfall[i, :] = np.mean(flag_array[uv.time_array == t, 0, :, :],
                                          axis=0)
        else:
            waterfall = np.zeros((uv.Ntimes, uv.Nfreqs))
            for i, t in enumerate(np.unique(uv.time_array)):
                waterfall[i, :] = np.mean(flag_array[uv.time_array == t, 0, :, :],
                                          axis=(0, 2))

    return waterfall


def and_rows_cols(waterfall):
    """ For a 2D flag waterfall, flag pixels only if fully flagged along
    time and/or frequency
    Args:
        waterfall - 2D boolean array of shape (Ntimes, Nfreqs)
    Returns:
        wf (2D array): A 2D array (size same as input) where only times/integrations
            that were fully flagged are flagged.
    """
    wf = np.zeros_like(waterfall, dtype=np.bool)
    Ntimes, Nfreqs = waterfall.shape
    wf[:, (np.sum(waterfall, axis=0) / Ntimes) == 1] = True
    wf[(np.sum(waterfall, axis=1) / Nfreqs) == 1] = True
    return wf


def lst_from_uv(uv):
    ''' Calculate the lst_array for a UVData or UVCal object.
    Args:
        uv: a UVData or UVCal object.
    Returns:
        lst_array: lst_array corresponding to time_array and at telescope location.
                   Units are radian.
    '''
    if not isinstance(uv, (UVCal, UVData)):
        raise ValueError('Function lst_from_uv can only operate on '
                         'UVCal or UVData object.')

    tel = uvtel.get_telescope(uv.telescope_name)
    lat, lon, alt = tel.telescope_location_lat_lon_alt_degrees
    lst_array = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    return lst_array
