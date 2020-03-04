# Contributing to pyuvdata

Thank you for considering contributing to pyuvdata! It's our community of users and contributors that makes pyuvdata powerful and relevant.

pyuvdata is an open source project, driven by our community of contributors. There are many ways to contribute, including writing tutorials, improving the documentation, submitting bug reports and feature requests or writing code which can be incorporated into pyuvdata itself. Following the guidelines and patterns suggested below helps us maintain a high quality code base and review your contributions faster.

Please note we have a [code of conduct](../CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

## How to report a bug
First check the issues to see if your bug already exists. Feel free to comment on the existing issue to provide more context or just to note that it is affecting you as well. If your bug is not in the issue list, make a new issue.

When making an issue, try to provide as much context as possible including:

1. What version of python and pyuvdata are you using?
2. What operating system are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?
6. Any code to reproduce the bug (as minimal an example as possible)

If you're really inspired, you can make a pull request adding a test that fails because of the bug. This is likely to lead to the bug being fixed more quickly.

## How to suggest a feature or enhancement
First check the issues to see if your feature request already exists. Feel free to comment on the existing issue to provide more context or just to note that you would like to see the feature implemented as well. If your feature request is not in the issue list, make a new issue.

When making a feature request, try to provide as much context as possible. Feel free to include suggestions for implementations.

## Guidelines for contributing to the code

* Create issues for any major changes and enhancements that you wish to make. Discuss things transparently and get community feedback.
* Keep pull requests as small as possible. Ideally each pull request should implement ONE feature or bugfix. If you want to add or fix more than one thing, submit more than one pull request.
* Do not commit changes to files that are irrelevant to your feature or bugfix.
* Be aware that the pull request review process is not immediate, and is generally proportional to the size of the pull request.
* Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. See our [Code of Conduct](../CODE_OF_CONDUCT.md).

### Your First Contribution

Contributing for the first time can seem daunting, but we value contributions from our user community and we will do our best to help you through the process. Here’s some advice to help make your work on pyuvdata more useful and rewarding.

* Use issue labels to guide you
  - Unsure where to begin contributing to pyuvdata? You can start by looking through issues labeled `good first issue` and `help wanted` issues.

* Pick a subject area that you care about, that you are familiar with, or that you want to learn about
  - Some of the feature request issues involve file formats that the core developers may not be very familiar with. If you know about those formats, consider helping on those issues. If you're most interested, for example, in representations of beams for simulation, consider focusing on issues labeled with `beam`.

* Start small
  - It’s easier to get feedback on a little issue or pull request than on a big one.

* If you’re going to take on a big change, make sure that your idea has support first
  - This means getting someone else to confirm that a bug is real before you fix the issue, and ensuring that there’s consensus on a proposed feature before you work to implement it. Use the issue log to start conversations about major changes and enhancements.

* Be bold! Leave feedback!
  - Sometimes it can be scary to make new issues or comment on existing issues or pull requests, but contributions from the wider community are what ensure that pyuvdata serves the whole community as well as possible.

* Be rigorous
  - Our requirements on code style, testing and documentation are important. If you have questions about them or difficulty meeting them, please ask for help, we will do our best to support you. Your contributions will be reviewed and integrated much more quickly if your pull request meets the requirements.

If you are new to the GitHub or the pull request process you can start by taking a look at these tutorials:
http://makeapullrequest.com/ and http://www.firsttimersonly.com/. If you have more questions, feel free to ask for help, everyone is a beginner at first and all of us are still learning!

### Getting started

1. Create your own fork or branch of the code.
2. Do the changes in your fork or branch.
3. Follow the [Developer Installation](../README.md#developer-installation) instructions to ensure that you have all the required packages for testing your changes.
4. If you like the change and think the project could use it:
  - If you're fixing a bug, include a new test that breaks as a result of the bug (if possible).
  - Ensure that all your new code is covered by tests and that the existing tests pass. Tests can be run by running `pytest` in the top level `pyuvdata` directory. To run tests and automatically create a coverage report, run the script `test_coverage.sh` in the `scripts` directory. Coverage reports require the `pytest-cov` plug-in. Testing of `UVFlag` module requires the `pytest-cases` plug-in.
  - Ensure that your code meets the Black style guidelines. You can check that your code will pass our linting tests by running `pre-commit run -a`  in the top-level pyuvdata directory.
  - Ensure that you fully document any new features via docstrings and in the [tutorial](../docs/tutorial.rst)
  - You can see the full pull request checklist [here](PULL_REQUEST_TEMPLATE.md)

### Code review process

The core team looks at pull requests on a regular basis and tries to provide feedback as quickly as possible. Larger pull requests generally require more time for review and may require discussion among the core team.

# Community
In addition to conversations on github, we also communicate via Slack and on monthly telecons. We welcome contributors who want to join those discussions, [contact the RASG managers](mailto:rasgmanagers@gmail.com) if you'd like to be invited to join them.
