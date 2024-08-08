# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""History management utilities."""


def _check_history_version(history, version_string):
    """Check if version_string is present in history string."""
    return version_string.replace(" ", "") in history.replace("\n", "").replace(" ", "")


def _check_histories(history1, history2):
    """Check if two histories are the same."""
    return history1.replace("\n", "").replace(" ", "") == history2.replace(
        "\n", ""
    ).replace(" ", "")


def _combine_history_addition(history1, history2):
    """
    Find extra history to add to have minimal repeats.

    Parameters
    ----------
    history1 : str
        First history.
    history2 : str
        Second history

    Returns
    -------
    str
        Extra history to add to first history.

    """
    # first check if they're the same to avoid more complicated processing.
    if _check_histories(history1, history2):
        return None

    hist2_words = history2.split(" ")
    add_hist = ""
    test_hist1 = " " + history1 + " "
    for i, word in enumerate(hist2_words):
        if " " + word + " " not in test_hist1:
            add_hist += " " + word
            keep_going = i + 1 < len(hist2_words)
            while keep_going:
                if (hist2_words[i + 1] == " ") or (
                    " " + hist2_words[i + 1] + " " not in test_hist1
                ):
                    add_hist += " " + hist2_words[i + 1]
                    del hist2_words[i + 1]
                    keep_going = i + 1 < len(hist2_words)
                else:
                    keep_going = False

    if add_hist == "":
        add_hist = None
    return add_hist
