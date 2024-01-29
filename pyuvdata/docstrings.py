"""Functions for dealing with merging docstrings."""

import copy
import typing as tp
from collections import ChainMap
from inspect import Signature
from itertools import chain

from docstring_parser import (
    DocstringMeta,
    DocstringParam,
    DocstringStyle,
    RenderingStyle,
    compose,
    parse,
)
from docstring_parser.util import _Func


def combine_docstrings(
    *others: _Func,
    exclude: tp.Iterable[tp.Type[DocstringMeta]] = (),
    style: DocstringStyle = DocstringStyle.AUTO,
    rendering_style: RenderingStyle = RenderingStyle.COMPACT,
):
    """Combine docstrings from multiple sources programmatically.

    The guts of this function are taken from the `docstring_parser` package.
    What it does differently is that it notices if a kwargs item is present in
    the signature, and writes parameters in the subdocs to Other Parameters.

    A function decorator that parses the docstrings from `others`,
    programmatically combines them with the parsed docstring of the decorated
    function, and replaces the docstring of the decorated function with the
    composed result. Only parameters that are part of the decorated functions
    signature are included in the combined docstring. When multiple sources for
    a parameter or docstring metadata exists then the decorator will first
    default to the wrapped function's value (when available) and otherwise use
    the rightmost definition from ``others``.

    """

    def wrapper(func: _Func) -> _Func:
        sig = Signature.from_callable(func)

        comb_doc = parse(func.__doc__ or "", style=style)
        docs = [parse(other.__doc__ or "", style=style) for other in others] + [
            comb_doc
        ]
        params = dict(
            ChainMap(
                *(
                    {param.arg_name: param for param in doc.params}
                    for doc in reversed(docs)
                )
            )
        )

        for doc in reversed(docs):
            if not doc.short_description:
                continue
            comb_doc.short_description = doc.short_description
            comb_doc.blank_after_short_description = doc.blank_after_short_description
            break

        for doc in reversed(docs):
            if not doc.long_description:
                continue
            comb_doc.long_description = doc.long_description
            comb_doc.blank_after_long_description = doc.blank_after_long_description
            break

        combined = {}
        for doc in docs:
            metas = {}
            for meta in doc.meta:
                meta_type = type(meta)
                if meta_type in exclude:
                    continue
                metas.setdefault(meta_type, []).append(meta)
            for meta_type, meta in metas.items():
                combined[meta_type] = meta

        combined[DocstringParam] = [
            params[name] for name in sig.parameters if name in params
        ]

        other_params = []
        if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
            # We have **kwargs
            # We need to add all the parameters that are not in the signature
            # to the Other Parameters section
            for name, param in params.items():
                if name not in sig.parameters:
                    param.args[0] = "other_param"
                    other_params.append(param)

        comb_doc.meta = list(chain(*combined.values())) + other_params

        func.__doc__ = compose(comb_doc, rendering_style=rendering_style, style=style)
        return func

    return wrapper


def copy_replace_short_description(
    other: _Func,
    style: DocstringStyle = DocstringStyle.AUTO,
    rendering_style: RenderingStyle = RenderingStyle.COMPACT,
):
    """Copy the long description and parameters section(s) from another docstring."""

    def wrapper(func: _Func) -> _Func:
        this_doc = parse(func.__doc__ or "", style=style)
        other_doc = parse(other.__doc__ or "", style=style)

        new_doc = copy.deepcopy(other_doc)
        new_doc.short_description = this_doc.short_description

        func.__doc__ = compose(new_doc, rendering_style=rendering_style, style=style)
        return func

    return wrapper
