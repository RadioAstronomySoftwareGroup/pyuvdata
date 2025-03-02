"""
Format the readme.md file into the sphinx index.rst file.

"""

import inspect
import os

import pypandoc
from astropy.time import Time


def write_index_rst(readme_file=None, write_file=None):
    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out = (
        ".. pyuvdata documentation top-level file, created by\n"
        f"   make_index.py on {t.iso}\n\n"
    )

    if readme_file is None:
        main_path = os.path.dirname(
            os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        )
        readme_file = os.path.join(main_path, "README.md")

    readme_text = pypandoc.convert_file(readme_file, "rst")

    # convert relative links in readme to explicit links
    readme_text = readme_text.replace(
        "<docs/",
        "<https://github.com/RadioAstronomySoftwareGroup/pyuvdata/tree/main/docs/",
    )

    readme_text = readme_text.replace(
        "<.github/",
        "<https://github.com/RadioAstronomySoftwareGroup/pyuvdata/tree/main/.github/",
    )

    out += readme_text
    out += (
        "\n\nFurther Documentation\n====================================\n"
        ".. toctree::\n"
        "   :maxdepth: 1\n\n"
        "   tutorial\n"
        "   conventions\n"
        "   uvdata\n"
        "   uvcal\n"
        "   uvbeam\n"
        "   uvflag\n"
        "   telescope\n"
        "   analytic_beams\n"
        "   beam_interface\n"
        "   fast_uvh5_meta\n"
        "   fast_calh5_meta\n"
        "   utility_functions\n"
        "   developer_docs\n"
    )

    out.replace("\u2018", "'").replace("\u2019", "'").replace("\xa0", " ")

    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "index.rst")
    with open(write_file, "w") as F:
        F.write(out)
    print("wrote " + write_file)
