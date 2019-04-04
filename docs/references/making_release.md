## How to make a new release on PyPI and conda-forge

### PyPI
1. iterate the version number. Our version numbering approach is:
    1. The first number is for very large changes which might happen every couple of years (e.g. 1.0).
    2. The second number is for regular releases, our goal is to issue these ~3-4 times per year.
    3. The third number is for small patches to fix issues. These happen as needed to get critical fixes onto PyPI and conda-forge.
2. update the changelog to put all the unreleased changes under the new version (leaving the unreleased section empty).
2. Make a PR for the version change and get it accepted & merged. Only make a PyPI distribution from a clean master branch (no local changes).
3. make the distribution: python setup.py sdist
4. upload to test site: twine upload --repository testpypi dist/*
5. check that it looks good at https://test.pypi.org/project/pyuvdata
6. upload to real site: twine upload --repository pypi dist/*

### Conda (do this after the PyPI release)
1. Fork the feedstock repo to your personal github account.
2. Make a new branch on your fork for the changes
3. get the new SHA from pypi: Go to the PyPI file listing page, next to each file there's a little link labeled SHA256 that will copy the right value to your clipboard. Or you can just download the file yourself and run `openssl sha256` on it.
4. update recipe/meta.yaml: minimally update the version, build & SHA (if it’s a new version, reset the build to zero, otherwise bump the build number). Generally review the whole file for things that should change.

  **Note:** When the PyPI package is updated, a bot may make a PR that only changes the the version, build and SHA (plus maybe some re-rendering). If these are the only required changes, you can just accept the bot's PR. If there are other needed changes, comment on the PR to stop it being merged by conda-forge admins without the other fixes.

5. push your branch to github
6. open a  PR against the feedstock repo from your branch.
7. get a bot to automatically re-render the code by commenting on the PR with `@conda-forge-admin, please rerender`
8. fix any problems
