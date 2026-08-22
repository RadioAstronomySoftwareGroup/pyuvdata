#!/usr/bin/env python
"""
Download the pyuvdata test data into a shared pooch cache.

The wheel-building jobs run the test suite once per wheel, in an environment with a
cold cache, so each one re-downloads the whole test dataset from github. Running this
once on the runner and pointing the test environments at the result via the
PYUVDATA_DATA_DIR environment variable turns those N downloads into one.

Usage: python ci/prefetch_test_data.py <cache_dir>
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "pyuvdata"


def load_datasets_module():
    """
    Load pyuvdata.datasets without importing pyuvdata itself.

    Importing the real package pulls in the compiled extensions, which have not been
    built yet when this runs. datasets.py only needs DATA_PATH from the parent, so
    stub out just enough of the package for it to import.
    """
    pkg = types.ModuleType("pyuvdata")
    pkg.__path__ = [str(SRC)]
    data_pkg = types.ModuleType("pyuvdata.data")
    data_pkg.DATA_PATH = str(SRC / "data")
    sys.modules["pyuvdata"] = pkg
    sys.modules["pyuvdata.data"] = data_pkg

    spec = importlib.util.spec_from_file_location(
        "pyuvdata.datasets", SRC / "datasets.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pyuvdata.datasets"] = module
    spec.loader.exec_module(module)
    return module


def main():
    """Fetch every dataset in the registry into the requested cache directory."""
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <cache_dir>")

    # pooch reads this when the Pooch object is created, so it has to be set before
    # datasets.py is executed.
    cache_dir = Path(sys.argv[1]).resolve()
    os.environ["PYUVDATA_DATA_DIR"] = str(cache_dir)

    datasets = load_datasets_module()
    print(f"Caching {len(datasets.fetch_dict)} datasets into {datasets.pup.abspath}")

    for name in datasets.fetch_dict:
        datasets.fetch_data(name)

    print(f"Done. Cache is at {datasets.pup.abspath}")


if __name__ == "__main__":
    main()
