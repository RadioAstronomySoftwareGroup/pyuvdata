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

readme_text = pypandoc.convert_file('../readme.md', 'rst')
end_text = 'parameters description'
regex = re.compile(end_text.replace(' ', '\s+'))
loc = re.search(regex, readme_text).start()

out += readme_text[0:loc] + end_text + '.'
out += ('\n\nFurther Documentation\n====================================\n'
        '.. toctree::\n'
        '   :maxdepth: 1\n\n'
        '   tutorial\n'
        '   parameters\n'
        '   developer_docs\n')
F = open('index.rst', 'w')
F.write(out)
print "wrote index.rst"
