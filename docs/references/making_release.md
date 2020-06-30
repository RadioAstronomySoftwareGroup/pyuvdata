## How to make a new release on PyPI and conda-forge

### PyPI
1. Define the newest version number. Our version numbering approach is:
    - The first number is for very large changes which might happen every couple of years (e.g. 1.0).
    - The second number is for regular releases, our goal is to issue these ~3-4 times per year.
    - The third number is for small patches to fix issues. These happen as needed to get critical fixes onto PyPI and conda-forge.
2. Update the changelog to put all the unreleased changes under the new version
(leaving the unreleased section empty).
3. Make a PR for the version change and get it accepted & merged.
5. Iterate the version number as a new tag. This can be accomplished through
the online interface or via the cli with: `git tag <hash> vX.Y.Z` where the
hashed commit must be the merge commit from the previous PR.
6. Check that the publish to pypi github action (which automatically makes the
release to PyPI when the tag is created) doesn't error.

### Conda (do this after the PyPI release)

When the PyPI package is updated, a bot will probably make a PR that only changes
the version, build and SHA (plus maybe some re-rendering). If these are the only
required changes, you can just accept the bot's PR. If there are other needed
changes (e.g. dependencies), comment on the PR to stop it being merged by
conda-forge admins without the other fixes. At that point you can either modify
the bot's PR to make the fixes or make your own PR by hand (details below).

To do it fully by hand:

1. Fork the feedstock repo to your personal github account.
2. Make a new branch on your fork for the changes
3. get the new SHA from pypi: Go to the PyPI file listing page, next to each
file there's a little link labeled SHA256 that will copy the right value to your
clipboard. Or you can just download the file yourself and run `openssl sha256` on it.
4. update recipe/meta.yaml: minimally update the version, build & SHA (if it’s
a new version, reset the build to zero, otherwise bump the build number).
Generally review the whole file for things that should change.
5. push your branch to github
6. open a  PR against the feedstock repo from your branch.
7. get a bot to automatically re-render the code by commenting on the PR with
`@conda-forge-admin, please rerender`
8. fix any problems
