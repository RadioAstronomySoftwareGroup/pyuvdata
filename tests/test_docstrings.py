"""Tests of the docstrings module."""

from docstring_parser import DocstringReturns, DocstringStyle

from pyuvdata.docstrings import combine_docstrings


def test_docstrings_combine():
    def a(x, y):
        """Do a.

        Some long description.

        Parameters
        ----------
        x : int
            The x parameter.
        y : int
            The y parameter.

        Returns
        -------
        int
            The sum of x and y.
        """
        return x + y

    @combine_docstrings(a, style=DocstringStyle.NUMPYDOC)
    def b(x, y, z):
        return a(x, y) + z

    assert "Do a." in b.__doc__
    assert "Some long description." in b.__doc__
    assert "The x parameter." in b.__doc__
    assert "The y parameter." in b.__doc__
    assert "The z parameter." not in b.__doc__
    assert "The sum of x and y." in b.__doc__

    @combine_docstrings(a, style=DocstringStyle.NUMPYDOC)
    def c(x, y, z):
        """Do c.

        Parameters
        ----------
        z : int
            The z parameter.
        """
        return a(x, y) + z + 1

    assert "Do c." in c.__doc__
    assert "Do a." not in c.__doc__
    assert "Some long description." in c.__doc__
    assert "The x parameter." in c.__doc__
    assert "The y parameter." in c.__doc__
    assert "The z parameter." in c.__doc__

    @combine_docstrings(
        a, c, exclude=(DocstringReturns,), style=DocstringStyle.NUMPYDOC
    )
    def d(x, y, z):
        """Do d."""
        return 1

    assert "The sum of x and y." not in d.__doc__
