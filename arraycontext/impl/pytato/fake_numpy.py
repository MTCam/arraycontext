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

import numpy as np

from arraycontext.fake_numpy import (
        BaseFakeNumpyLinalgNamespace
        )
from arraycontext.loopy import (
        LoopyBasedFakeNumpyNamespace
        )
from arraycontext.container import NotAnArrayContainerError, serialize_container
from arraycontext.container.traversal import (
        rec_map_array_container,
        rec_multimap_array_container,
        rec_map_reduce_array_container,
        )
import pytato as pt


class PytatoFakeNumpyLinalgNamespace(BaseFakeNumpyLinalgNamespace):
    # Everything is implemented in the base class for now.
    pass


class PytatoFakeNumpyNamespace(LoopyBasedFakeNumpyNamespace):
    """
    A :mod:`numpy` mimic for :class:`PytatoPyOpenCLArrayContext`.

    .. note::

        :mod:`pytato` does not define any memory layout for its arrays. See
        :ref:`Pytato docs <pytato:memory-layout>` for more on this.
    """

    _pt_funcs = frozenset({
        "sin", "cos", "tan", "arcsin", "arccos", "arctan",
        "sinh", "cosh", "tanh", "exp", "log", "log10",
        "sqrt", "abs", "isnan"
        })

    def _get_fake_numpy_linalg_namespace(self):
        return PytatoFakeNumpyLinalgNamespace(self._array_context)

    def __getattr__(self, name):
        if name in self._pt_funcs:
            from functools import partial
            return partial(rec_map_array_container, getattr(pt, name))

        return super().__getattr__(name)

    # NOTE: the order of these follows the order in numpy docs
    # NOTE: when adding a function here, also add it to `array_context.rst` docs!

    # {{{ array creation routines

    def ones_like(self, ary):
        return self.full_like(ary, 1)

    def full_like(self, ary, fill_value):
        def _full_like(subary):
            return pt.full(subary.shape, fill_value, subary.dtype)

        return self._new_like(ary, _full_like)

    # }}}

    # {{{ array manipulation routines

    def reshape(self, a, newshape, order="C"):
        return rec_map_array_container(
                lambda ary: pt.reshape(a, newshape, order=order),
                a)

    def ravel(self, a, order="C"):
        """
        :arg order: A :class:`str` describing the order in which the elements
            must be traversed while flattening. Can be one of 'F', 'C', 'A' or
            'K'. Since, :mod:`pytato` arrays don't have a memory layout, if
            *order* is 'A' or 'K', the traversal order while flattening is
            undefined.
        """

        def _rec_ravel(a):
            if order in "FC":
                return pt.reshape(a, (-1,), order=order)
            elif order in "AK":
                # flattening in a C-order
                # memory layout is assumed to be "C"
                return pt.reshape(a, (-1,), order="C")
            else:
                raise ValueError("`order` can be one of 'F', 'C', 'A' or 'K'. "
                                 f"(got {order})")

        return rec_map_array_container(_rec_ravel, a)

    def transpose(self, a, axes=None):
        return rec_multimap_array_container(pt.transpose, a, axes)

    def broadcast_to(self, array, shape):
        return rec_map_array_container(partial(pt.broadcast_to, shape=shape), array)

    def concatenate(self, arrays, axis=0):
        return rec_multimap_array_container(pt.concatenate, arrays, axis)

    def stack(self, arrays, axis=0):
        return rec_multimap_array_container(
                lambda *args: pt.stack(arrays=args, axis=axis),
                *arrays)

    # }}}

    # {{{ logic functions

    def all(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, pt.logical_and),
                lambda subary: pt.all(subary), a)

    def any(self, a):
        return rec_map_reduce_array_container(
                partial(reduce, pt.logical_or),
                lambda subary: pt.any(subary), a)

    def array_equal(self, a, b):
        actx = self._array_context

        # NOTE: not all backends support `bool` properly, so use `int8` instead
        false = actx.from_numpy(np.int8(False))

        def rec_equal(x, y):
            if type(x) != type(y):
                return false

            try:
                iterable = zip(serialize_container(x), serialize_container(y))
            except NotAnArrayContainerError:
                if x.shape != y.shape:
                    return false
                else:
                    return pt.all(pt.equal(x, y))
            else:
                return reduce(
                        pt.logical_and,
                        [rec_equal(ix, iy) for (_, ix), (_, iy) in iterable]
                        )

        return rec_equal(a, b)

    def greater(self, x, y):
        return rec_multimap_array_container(pt.greater, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(pt.greater_equal, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(pt.less, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(pt.less_equal, x, y)

    def equal(self, x, y):
        return rec_multimap_array_container(pt.equal, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(pt.not_equal, x, y)

    # }}}

    # {{{ mathematical functions

    def arctan2(self, y, x):
        return rec_multimap_array_container(pt.arctan2, y, x)

    def sum(self, a, axis=None, dtype=None):
        def _pt_sum(ary):
            if dtype not in [ary.dtype, None]:
                raise NotImplementedError

            return pt.sum(ary, axis=axis)

        return rec_map_reduce_array_container(sum, _pt_sum, a)

    def conj(self, x):
        return rec_multimap_array_container(pt.conj, x)

    def maximum(self, x, y):
        return rec_multimap_array_container(pt.maximum, x, y)

    def amax(self, a, axis=None):
        return rec_map_reduce_array_container(
                partial(reduce, pt.maximum), partial(pt.amax, axis=axis), a)

    max = amax

    def minimum(self, x, y):
        return rec_multimap_array_container(pt.minimum, x, y)

    def amin(self, a, axis=None):
        return rec_map_reduce_array_container(
                partial(reduce, pt.minimum), partial(pt.amin, axis=axis), a)

    min = amin

    def absolute(self, a):
        return self.abs(a)

    # }}}

    # {{{ sorting, searching, and counting

    def where(self, criterion, then, else_):
        return rec_multimap_array_container(pt.where, criterion, then, else_)

    # }}}
