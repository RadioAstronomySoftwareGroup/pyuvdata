# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Test data setup with Pooch."""

import os
import warnings

import pooch
import yaml
from pooch import Untar

from pyuvdata.data import DATA_PATH

# set the data version. This should be updated when the test data are changed.
data_version = "v0.0.4"

pup = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("pyuvdata"),
    # The remote data is on Github
    base_url="https://github.com/RadioAstronomySoftwareGroup/rasg-datasets/raw/{version}",
    version=data_version,
    # If this is a development version, get the data from the "main" branch
    version_dev="main",
    registry=None,
)

# Get registry file from package_data
registry_file = os.path.join(DATA_PATH, "test_data_registry.txt")
# Load this registry file
pup.load_registry(registry_file)

test_data_yaml = os.path.join(DATA_PATH, "test_data.yaml")
with open(test_data_yaml) as file:
    fetch_dict = yaml.safe_load(file)


def fetch_data(name: str | list):
    """
    Fetch a dataset by key in the fetch_dict.

    Parameters
    ----------
    name : str or list of str
        Name of the dataset (key in fetch_dict).

    Returns
    -------
    filename : str or list of str
        Location of the file(s) or folder on the local computer.
    """
    input_list = True
    if isinstance(name, str):
        input_list = False
        name = [name]
    fname = []
    for dname in name:
        # The data will be downloaded automatically the first time this is run for it.
        # Afterwards, Pooch finds it in the local cache and doesn't repeat the download.
        reg_name = fetch_dict[dname]
        if ".tar" in reg_name:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Python 3.14 will, by default, filter extracted "
                    "tar archives",
                )
                pup.fetch(reg_name, processor=Untar(extract_dir="."))
            folder_name = str(pup.abspath / reg_name.split(".tar")[0])
            fname.append(folder_name)
        else:
            fname.append(pup.fetch(reg_name))
    if len(fname) == 1 and not input_list:
        return fname[0]
    return fname
