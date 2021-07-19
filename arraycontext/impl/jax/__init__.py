"""
.. currentmodule:: arraycontext
.. autoclass:: JAXArrayContext
"""

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

from typing import Sequence, Union, Callable, Any

import numpy as np

from pytools.tag import Tag

from arraycontext.context import ArrayContext


class JAXArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :mod:`jax.numpy.DeviceArray` instances
    for its base array class.
    """

    def __init__(self) -> None:
        r"""
        .. note::

            JAX stores a global configuration state in the module
            :mod:`jax.config`. Callers are expected to maintain those. Most
            important for scientific computing workloads being
            ``jax_enable_x64``.
        """
        super().__init__()

        from jax.numpy import DeviceArray
        self.array_types = (DeviceArray,)

    def _get_fake_numpy_namespace(self):
        from .fake_numpy import JAXFakeNumpyNamespace
        return JAXFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        import jax.numpy as jnp
        return jnp.empty(shape=shape, dtype=dtype)

    def zeros(self, shape, dtype):
        import jax.numpy as jnp
        return jnp.zeros(shape=shape, dtype=dtype)

    def from_numpy(self, array: np.ndarray):
        import jax
        return jax.device_put(array)

    def to_numpy(self, array):
        import jax
        return jax.device_get(array)

    def call_loopy(self, t_unit, **kwargs):
        from .utils import loopy_to_jax
        jax_fn = loopy_to_jax(t_unit)
        return jax_fn(**kwargs)

    def freeze(self, array):
        return array.block_until_ready()

    def thaw(self, array):
        return array

    # }}}

    def tag(self, tags: Union[Sequence[Tag], Tag], array):
        # Sorry, not capable.
        return array

    def tag_axis(self, iaxis, tags: Union[Sequence[Tag], Tag], array):
        # TODO: See `jax.experiemental.maps.xmap`, proabably that should be useful
        return array

    def clone(self):
        return type(self)()

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        from jax import jit
        return jit(f)

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        import jax.numpy as jnp
        if arg_names is not None:
            from warnings import warn
            warn("'arg_names' don't bear any significance in "
                 "JAXArrayContext.", stacklevel=2)

        # TODO: tags
        return jnp.einsum(spec, *args)

    @property
    def permits_inplace_modification(self):
        return False

# vim: foldmethod=marker
