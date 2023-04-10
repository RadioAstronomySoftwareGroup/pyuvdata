# How to make a new release on GitHub, PyPI and conda-forge

## Making the release (tag) on GitHub

1. Define the newest version number. Our version numbering approach is:
    - The first number is for very large changes which might happen every couple of years (e.g. 1.0).
    - The second number is for regular releases, our goal is to issue these ~3-4 times per year.
    - The third number is for small patches to fix issues. These happen as needed to
    get critical fixes onto PyPI and conda-forge.
2. Update the changelog to put all the unreleased changes under the new version
(leaving the unreleased section empty).
3. Make a PR for the version change and get it accepted & merged.
4. Iterate the version number as a new tag. This can be accomplished through
the online interface (this is the best approach) or via the cli with:
`git tag <hash> vX.Y.Z` where the hashed commit must be the merge commit from the previous PR.

## PyPI (do this after the GitHub release)

When the tag is made on GitHub, a GitHub Actions workflow automatically publishes the
new release to PyPI. Check that the publish to pypi github action doesn't error. If it
does error, follow the steps below:

To release to PyPI by hand:
1. Checkout the main branch and pull. Ensure that there are no local changes. PyPI
distributions should only be made from a clean main branch!
1. make the distribution: ```python -m build```
2. upload to test site: twine upload --repository testpypi dist/*
3. check that it looks good at https://test.pypi.org/project/pyuvdata
4. upload to real site: twine upload --repository pypi dist/*

## Conda (do this after the PyPI release)

When the PyPI package is updated, a bot will probably make a PR that only changes
the version, build and SHA (plus maybe some re-rendering). If these are the only
required changes, you can just accept the bot's PR. If there are other needed
changes (e.g. dependencies), comment on the PR to stop it being merged by
conda-forge admins without the other fixes. At that point you can either modify
the bot's PR to make the fixes or make your own PR by hand (details below).

To do it fully by hand:

1. Fork the feedstock repo to your personal github account.
2. Make a new branch on your fork for the changes
3. get the new SHA from pypi: Go to the PyPI download page for this release. Next to the
source distribution (.tar.gz file) there's a "view hashes" link. Click that link to see
the SHA256 hash and click the "copy" link next to it to copy the right value to your
clipboard. Or you can just download the file yourself and run `openssl sha256` on it.
4. update recipe/meta.yaml: minimally update the version, build & SHA (if it’s
a new version, reset the build to zero, otherwise bump the build number).
Generally review the whole file for things that should change, particularly any
dependency changes.
5. push your branch to github
6. open a  PR against the feedstock repo from your branch.
7. get a bot to automatically re-render the code by commenting on the PR with
`@conda-forge-admin, please rerender`
8. fix any problems
