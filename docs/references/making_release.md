How to make a new release on PyPI and conda-forge

PyPI:
1. iterate the version number
2. make the distribution: python setup.py sdist
3. upload to test site: twine upload --repository testpypi dist/*
4. check that it looks good at https://test.pypi.org/
5. upload to real site: twine upload --repository pypi dist/*

Conda:
1. Fork the feedstock repo
2. Make a new branch on your fork for the changes
3. get the new SHA from pypi: Go to the PyPI file listing page, next to each file there's a little link labeled SHA256 that will copy the right value to your clipboard. Or you can just download the file yourself and run `openssl sha256` on it.
4. update recipe/meta.yaml: minimally update the version, build & SHA (ff it’s a new version, reset the build to zero, otherwise bump the build number), generally review the whole file for things that should change. (If these are the only changes, a bot may make a PR that does all this for you. If there are other needed changes and the bot makes a PR without those changes, comment on the PR to stop it being merged without the other fixes.)
5. push your branch to github
6. open a  PR against the feedstock repo from your branch.
7. get a bot to automatically re-render the code by commenting on the PR with `@conda-forge-admin, please rerender`
8. fix any problems
