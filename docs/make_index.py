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
travis_str = 'https://travis-ci.org/HERA-Team/pyuvdata.svg'
regex_travis = re.compile(travis_str)
loc_travis_start = re.search(regex_travis, readme_md).start()
loc_travis_end = re.search(regex_travis, readme_md).end()
end_branch_str = '\)\]'
regex_end = re.compile(end_branch_str)
loc_branch_end = re.search(regex_end, readme_md).start()
branch_str = readme_md[loc_travis_end:loc_branch_end]

cover_str = 'https://coveralls.io/repos/github/HERA-Team/pyuvdata/badge.svg'
regex_cover = re.compile(cover_str)
loc_cover_start = re.search(regex_cover, readme_md).start()
loc_cover_end = re.search(regex_cover, readme_md).end()

readme_text = pypandoc.convert_file('../readme.md', 'rst')

rst_status_badge = '.. image:: ' + travis_str + branch_str + '\n    :target: https://travis-ci.org/HERA-Team/pyuvdata'
status_badge_text = '|Build Status|'
readme_text = readme_text.replace(status_badge_text, rst_status_badge + '\n\n')

rst_status_badge = '.. image:: ' + cover_str + branch_str + '\n    :target: https://coveralls.io/github/HERA-Team/pyuvdata' + branch_str
status_badge_text = '|Coverage Status|'
readme_text = readme_text.replace(status_badge_text, rst_status_badge)

readme_text = readme_text.replace(' ' + rst_status_badge, rst_status_badge)

end_text = 'parameters descriptions'
regex = re.compile(end_text.replace(' ', '\s+'))
loc = re.search(regex, readme_text).start()

out += readme_text[0:loc] + end_text + '.'
out += ('\n\nFurther Documentation\n====================================\n'
        '.. toctree::\n'
        '   :maxdepth: 1\n\n'
        '   tutorial\n'
        '   uvdata_parameters\n'
        '   uvdata\n'
        '   uvcal_parameters\n'
        '   uvcal\n'
        '   developer_docs\n')
F = open('index.rst', 'w')
F.write(out)
print("wrote index.rst")
