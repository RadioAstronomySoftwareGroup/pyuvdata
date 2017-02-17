"""
Format the readme.md file into the sphinx index.rst file.
"""
import re
import pypandoc
from astropy.time import Time

t = Time.now()
t.out_subfmt = 'date'
out = ('.. pyuvdata documentation master file, created by\n'
       '   make_index.py on {date}\n\n').format(date=t.iso)

readme_md = pypandoc.convert_file('../readme.md', 'md')
branch_str = 'https://travis-ci.org/HERA-Team/pyuvdata.svg'
regex_branch = re.compile(branch_str)
loc_branch = re.search(regex_branch, readme_md).start()
end_branch_str = '\)\]'
regex_end = re.compile(end_branch_str)
loc_end = re.search(regex_end, readme_md).start()
branch_url = readme_md[loc_branch:loc_end]

readme_text = pypandoc.convert_file('../readme.md', 'rst')

rst_status_badge = '.. image:: ' + branch_url + '\n    :target: https://travis-ci.org/HERA-Team/pyuvdata'
status_badge_text = '|Build Status|'
readme_text = readme_text.replace(status_badge_text, rst_status_badge)

end_text = 'parameters description'
regex = re.compile(end_text.replace(' ', '\s+'))
loc = re.search(regex, readme_text).start()

out += readme_text[0:loc] + end_text + '.'
out += ('\n\nFurther Documentation\n====================================\n'
        '.. toctree::\n'
        '   :maxdepth: 1\n\n'
        '   tutorial\n'
        '   parameters\n'
        '   uvdata\n'
        '   developer_docs\n')
F = open('index.rst', 'w')
F.write(out)
print("wrote index.rst")
