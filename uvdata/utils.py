import numpy as np
import warnings
import collections
import sys

# parameters for transforming between xyz & lat/lon/alt
gps_b = 6356752.31424518
gps_a = 6378137
e_squared = 6.69437999014e-3
e_prime_squared = 6.73949674228e-3


def LatLonAlt_from_XYZ(xyz):

        # see wikipedia geodetic_datum and Datum transformations of
        # GPS positions PDF in docs folder
        gps_p = np.sqrt(xyz[0]**2 + xyz[1]**2)
        gps_theta = np.arctan2(xyz[2] * gps_a, gps_p * gps_b)
        latitude = np.arctan2(xyz[2] + e_prime_squared * gps_b *
                              np.sin(gps_theta)**3, gps_p - e_squared * gps_a *
                              np.cos(gps_theta)**3)

        longitude = np.arctan2(xyz[1], xyz[0])
        gps_N = gps_a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
        altitude = ((gps_p / np.cos(latitude)) - gps_N)
        return latitude, longitude, altitude


def XYZ_from_LatLonAlt(latitude, longitude, altitude):

        # see wikipedia geodetic_datum and Datum transformations of
        # GPS positions PDF in docs folder
        gps_N = gps_a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
        xyz = np.zeros(3)
        xyz[0] = ((gps_N + altitude) * np.cos(latitude) * np.cos(longitude))
        xyz[1] = ((gps_N + altitude) * np.cos(latitude) * np.sin(longitude))
        xyz[2] = ((gps_b**2 / gps_a**2 * gps_N + altitude) * np.sin(latitude))

        return xyz


# Functions that are useful for testing:

def get_iterable(x):
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)


def clearWarnings():
    # Quick code to make warnings reproducible
    for name, mod in list(sys.modules.items()):
        try:
            reg = getattr(mod, "__warningregistry__", None)
        except ImportError:
            continue
        if reg:
            reg.clear()


def checkWarnings(func, func_args=[], func_kwargs={},
                  category=UserWarning,
                  nwarnings=1, message=None, known_warning=None):

    if known_warning == 'miriad':
        # The default warnings for known telescopes when reading miriad files
        category = [UserWarning]
        message = ['Altitude is not present in Miriad file, using known '
                   'location values for PAPER.']
        nwarnings = 1
    elif known_warning == 'paper_uvfits':
        # The default warnings for known telescopes when reading uvfits files
        category = [UserWarning] * 2
        message = ['Required Antenna frame keyword', 'telescope_location is not set']
        nwarnings = 2

    category = get_iterable(category)
    message = get_iterable(message)

    clearWarnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # All warnings triggered
        output = func(*func_args, **func_kwargs)  # Run function
        # Verify
        status = True
        if len(w) != nwarnings:
            print('wrong number of warnings')
            for idx, wi in enumerate(w):
                print('warning {i} is: {w}'.format(i=idx, w=wi))
            status = False
        else:
            for i, w_i in enumerate(w):
                if w_i.category is not category[i]:
                    status = False
                if message[i] is not None:
                    if message[i] not in str(w_i.message):
                        status = False
    return output, status
