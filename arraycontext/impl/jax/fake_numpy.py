__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from functools import partial, reduce

from arraycontext.fake_numpy import (
        BaseFakeNumpyNamespace, BaseFakeNumpyLinalgNamespace,
        )
from arraycontext.container.traversal import (
        rec_multimap_array_container, rec_map_array_container,
        rec_map_reduce_array_container,
        )
import jax.numpy as jnp


class JAXFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


class JAXFakeNumpyNamespace(BaseFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`JAXArrayContext`.
    """
    def _get_fake_numpy_linalg_namespace(self):
        return JAXFakeNumpyLinalgNamespace(self._array_context)

    def __getattr__(self, name):

        pt_funcs = ["abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
                    "sinh", "cosh", "tanh", "exp", "log", "log10", "isnan",
                    "sqrt", "exp", "conj"]
        if name in pt_funcs:
            from functools import partial
            return partial(rec_map_array_container, getattr(jnp, name))

        return super().__getattr__(name)

    def reshape(self, a, newshape):
        return rec_multimap_array_container(jnp.reshape, a, newshape)

    def transpose(self, a, axes=None):
        return rec_multimap_array_container(jnp.transpose, a, axes)

    def concatenate(self, arrays, axis=0):
        return rec_multimap_array_container(jnp.concatenate, arrays, axis)

    def ones_like(self, ary):
        return jnp.ones_like(ary)

    def maximum(self, x, y):
        return rec_multimap_array_container(jnp.maximum, x, y)

    def minimum(self, x, y):
        return rec_multimap_array_container(jnp.minimum, x, y)

    def where(self, criterion, then, else_):
        return rec_multimap_array_container(jnp.where, criterion, then, else_)

    def sum(self, a, dtype=None):
        def _pt_sum(ary):
            return jnp.sum(ary)

        return rec_map_reduce_array_container(sum, _pt_sum, a)

    def min(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, jnp.minimum), jnp.amin, a)

    def max(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, jnp.maximum), jnp.amax, a)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: jnp.stack(arrays=args, axis=axis),
                *arrays)

    # {{{ relational operators

    def equal(self, x, y):
        return rec_multimap_array_container(jnp.equal, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(jnp.not_equal, x, y)

    def greater(self, x, y):
        return rec_multimap_array_container(jnp.greater, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(jnp.greater_equal, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(jnp.less, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(jnp.less_equal, x, y)

    def arctan2(self, y, x):
        return rec_multimap_array_container(jnp.arctan2, y, x)

    def ravel(self, a, order="C"):
        def _rec_ravel(a):
            return jnp.ravel(a, order=order)

        return rec_map_array_container(_rec_ravel, a)

    # }}}
