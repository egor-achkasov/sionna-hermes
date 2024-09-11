#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Classes and functions relating to reconfigurable intelligent surfaces
"""

from abc import ABC
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from .radio_device import RadioDevice
from .scene_object import SceneObject
from . import scene
from .utils import rotate, normalize, outer


class CellGrid:
    # pylint: disable=line-too-long
    r"""
    Class defining a cell grid that determines the physical structure of a RIS

    The cell grid specifies the location of unit cells within the y-z plane
    assuming a homogenous spacing of 0.5. The actual positions are computed by
    multiplying the cell positions by the wavelength and rotating them
    according to the RIS' orientation.

    A cell grid must have at least three columns and rows to ensure
    that discrete phase and amplitude profiles of the RIS can be interpolated.

    Parameters
    ----------
    num_rows : int
        Number of rows. Must at least be equal to three.

    num_cols : int
        Number of columns. Must at least be equal to three.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.

    """

    def __init__(self, num_rows, num_cols, dtype=np.complex_):

        if dtype not in (np.complex_, np.complex_):
            raise ValueError("`dtype` must be np.complex_ or np.complex_`")
        self._dtype = dtype
        self._rdtype = np.float_

        if num_rows < 3 or num_cols < 3:
            raise ValueError("num_rows and num_cols must be >= 3")
        self._num_rows = int(num_rows)
        self._num_cols = int(num_cols)

        self._cell_y_positions = np.arange(self.num_cols, dtype=self._rdtype)
        self._cell_y_positions -= (self.num_cols - 1.0) / 2.0

        self._cell_z_positions = np.arange(self.num_rows - 1, -1, -1, dtype=self._rdtype)
        self._cell_z_positions -= (self.num_rows - 1.0) / 2.0

        z, y = np.meshgrid(self.cell_z_positions, self.cell_y_positions)
        self._cell_positions = np.stack([np.reshape(y, [-1]), np.reshape(z, [-1])], -1)

    @property
    def num_rows(self):
        r"""
        int : Number of rows
        """
        return self._num_rows

    @property
    def num_cols(self):
        r"""
        int : Number of columns
        """
        return self._num_cols

    @property
    def num_cells(self):
        r"""
        int : Number of cells
        """
        return self.num_rows * self.num_cols

    @property
    def cell_positions(self):
        r"""
        [num_cells, 2], np.float_ : Cell positions ordered from
            top-to-bottom left-to-right
        """
        return self._cell_positions

    @property
    def cell_y_positions(self):
        r"""
        [num_cols], np.float_ : y-coordinates of cells ordered
            from left-to-right
        """
        return self._cell_y_positions

    @property
    def cell_z_positions(self):
        r"""
        [num_rows], np.float_ : z-coordinates of cells ordered
            from top-to-bottom
        """
        return self._cell_z_positions


class Profile(ABC):
    # pylint: disable=line-too-long
    r"""Abstract class defining a phase/amplitude profile of a RIS

    A Profile instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.

    Input
    -----
    points : np.float_, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], np.float_
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """

    def __init__(self, dtype=np.complex_):
        self._dtype = dtype
        self._rdtype = np.float_

    @property
    @abstractmethod
    def num_modes(self):
        r"""
        int : Number of reradiation modes
        """
        pass

    @abstractmethod
    def __call__(self, points, mode=None, return_grads=False):
        r"""
        Returns the profile values, gradients and Hessians at given points

        Input
        -----
        points : np.float_, [num_samples, 2]
            Tensor of 2D coordinates defining the points on the RIS at which
            the profile should be evaluated.
            Defaults to `None`. In this case, the values for all unit cells
            are returned.

        mode : int | `None`
            Reradiation mode to be considered.
            Defaults to `None`. In this case, the values for all modes
            are returned.

        return_grads : bool
            If `True`, also the first- and second-order derivatives are
            returned.
            Defaults to `False`.

        Output
        ------
        values : [num_modes, num_samples] or [num_samples], np.float_
            Interpolated profile values at the sample positions

        grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
            Gradients of the interpolated profile values
            at the sample positions. Only returned if `return_grads` is `True`.

        hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
            Hessians of the interpolated profile values
            at the sample positions. Only returned if `return_grads` is `True`.
        """
        pass


class AmplitudeProfile(Profile):
    # pylint: disable=line-too-long
    r"""Abstract class defining an amplitude profile of a RIS

    An AmplitudeProfile instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.

    Input
    -----
    points : np.float_, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], np.float_
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """

    @property
    @abstractmethod
    def mode_powers(self):
        r"""
        [num_modes], np.float_: Relative power of reradiation modes
        """
        pass


class PhaseProfile(Profile):
    # pylint: disable=line-too-long
    r"""Abstract class defining a phase profile of a RIS

    A PhaseProfile instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.

    Input
    -----
    points : np.float_, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], np.float_
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """
    pass


class DiscreteProfile(Profile):
    # pylint: disable=line-too-long
    r"""Class defining a discrete phase/amplitude profile of a RIS

    A DiscreteProfile instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    cell_grid : :class:`~sionna.rt.CellGrid`
        Defines the physical structure of the RIS

    num_modes : int
        Number of reradiation modes.
        Defaults to 1.

    values : np.float_ or tf.Variable, [num_modes, num_rows, num_cols]
        Values of the discrete profile for each reradiation mode
        and unit cell. `num_rows` and `num_cols` are defined by the
        `cell_grid`.
        Defaults to `None`.

    interpolator : :class:`~sionna.rt.ProfileInterpolator`
        Instance of a `ProfileInterpolator` that interpolates the
        discrete values of the profile to a continuous profile
        which is defined at any point on the RIS.
        Defaults to `None`. In this case, the
        :class:`~sionna.rt.LagrangeProfileInterpolator` will be used.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.

    Input
    -----
    points : np.float_, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], np.float_
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """

    def __init__(
        self, cell_grid, num_modes=1, values=None, interpolator=None, dtype=np.complex_
    ):

        super().__init__(dtype=dtype)
        self._cell_grid = cell_grid
        self._num_modes = num_modes
        if values is None:
            self._values = None
        else:
            self.values = values
        if interpolator is None:
            self._interpolator = LagrangeProfileInterpolator(self)
        else:
            self._interpolator = interpolator

    @property
    def shape(self):
        r"""
        tf.TensorShape : Shape of the tensor holding the values of
            the discrete profile
        """
        return (self.num_modes, self.cell_grid.num_rows, self.cell_grid.num_cols)

    @property
    def values(self):
        r"""
        [shape], np.float_ : Set/get the discrete values of the profile for each
            reradiation mode
        """
        return self._values

    @values.setter
    def values(self, v):
        if not v.shape == self.shape:
            raise ValueError(f"`values` must have shape {self.shape}")
        if isinstance(v, np.ndarray):
            if v.dtype != self._rdtype:
                msg = f"`values` must have dtype={self._rdtype}"
                raise TypeError(msg)
            else:
                self._values = v
        else:
            self._values = np.ndarray(v, dtype=self._rdtype)

    @property
    def num_modes(self):
        r"""
        int : Number of reradiation modes
        """
        return self._num_modes

    @property
    def cell_grid(self):
        r"""
        :class:`~sionna.rt.CellGrid` : Defines the physical
            structure of the RIS
        """
        return self._cell_grid

    @property
    def spacing(self):
        r"""
        np.float_: Element spacing [m] corresponding to
            half a wavelength
        """
        if hasattr(scene.Scene(), "wavelength"):
            wavelength = scene.Scene().wavelength
            return wavelength / 2.
        else:
            # Scene is not initialized
            return 0.5

    def show(self, mode=0):
        r"""Visualizes the profile as a 3D plot

        Input
        ------
        mode : int | `None`
            Reradation mode to be shown.
            Defaults to 0.

        Output
        ------
        : :class:`matplotlib.pyplot.Figure`
            3D plot of the profile
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        y, z = np.meshgrid(
            self.cell_grid.cell_y_positions * self.spacing,
            self.cell_grid.cell_z_positions * self.spacing,
        )
        ax.plot_surface(y, z, self.values[mode], cmap="viridis")
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        if isinstance(self, PhaseProfile):
            plt.title(r"Phase profile $\chi(y, z)$")
        if isinstance(self, AmplitudeProfile):
            plt.title(r"Amplitude profile $A(y, z)$")
        return fig

    def __call__(self, points=None, mode=None, return_grads=False):
        r"""
        Returns the profile values, gradients and Hessians at given points

        Input
        -----
        points : np.float_, [num_samples, 2]
            Tensor of 2D coordinates defining the points on the RIS at which
            the profile should be evaluated.
            Defaults to `None`. In this case, the values for all unit cells
            are returned.

        mode : int | `None`
            Reradiation mode to be considered.
            Defaults to `None`. In this case, the values for all modes
            are returned.

        return_grads : bool
            If `True`, also the first- and second-order derivatives are
            returned. Only available if `points` is not `None`.
            Defaults to `False`.

        Output
        ------
        values : [num_modes, num_samples] or [num_samples], np.float_
            Interpolated profile values at the sample positions

        grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
            Gradients of the interpolated profile values
            at the sample positions. Only returned if `return_grads` is `True`.

        hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
            Hessians of the interpolated profile values
            at the sample positions. Only returned if `return_grads` is `True`.
        """
        if points is None:
            if mode is not None:
                values = np.transpose(self.values[mode])
                values = np.reshape(values, [-1])
            else:
                values = np.transpose(self.values, perm=[0, 2, 1])
                values = np.reshape(values, [self.num_modes, -1])
            return values
        else:
            return self._interpolator(points, mode, return_grads)


class ProfileInterpolator(ABC):
    r"""
    Abstract class defining an interpolator of a discrete profile

    A ProfileInterpolator instance is a callable that interpolate
    the discrete profile to specified points. Optionally, the
    gradients and Hessians are returned.

    Parameters
    ----------
    discrete_profile : :class:`~sionna.rt.DiscreteProfile`
        Discrete profile to be interpolated

    Input
    -----
    points : [num_samples, 2], np.float_
        Positions at which to interpolate the profile

    mode : int | `None`
        Mode of the profile to interpolate. If `None`.
        all modes are interpolated.
        Defaults to `None`.

    return_grads : bool
        If `True`, gradients and Hessians are computed.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], np.float_
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated profile values
        at the sample positions

    hessians : [num_modes, num_samples, 3, 3] or [num_samples,3,3], np.float_
        Hessians of the interpolated profile values
        at the sample positions
    """

    def __init__(self, discrete_profile):
        self._discrete_profile = discrete_profile
        self._dtype = discrete_profile._dtype
        self._rdtype = discrete_profile._rdtype

    @property
    def spacing(self):
        r"""
        np.float_: Element spacing [m] corresponding to
            half a wavelength
        """
        if hasattr(scene.Scene(), "wavelength"):
            wavelength = scene.Scene().wavelength
            return wavelength / 2.
        else:
            # Scene is not initialized
            return 0.5

    @property
    def cell_y_positions(self):
        r"""
        [num_cols], np.float_ : y-coordinates of cells ordered
            from left-to-right
        """
        return self._discrete_profile.cell_grid.cell_y_positions * self.spacing

    @property
    def cell_z_positions(self):
        r"""
        [num_rows], np.float_ : z-coordinates of cells ordered
            from top-to-bottom
        """
        return self._discrete_profile.cell_grid.cell_z_positions * self.spacing

    @property
    def num_rows(self):
        r"""
        int : Number of rows
        """
        return self._discrete_profile.cell_grid.num_rows

    @property
    def num_cols(self):
        r"""
        int : Number of columns
        """
        return self._discrete_profile.cell_grid.num_cols

    @property
    def values(self):
        r"""
        [shape], np.float_ : Discrete values of the profile for each
            reradiation mode and unit cell
        """
        return self._discrete_profile.values

    @abstractmethod
    def __call__(self, points, mode=None, return_grads=False):
        r"""
        Interpolates the discrete profile to specified points

        Optionally, the gradients and Hessians are returned.

        Input
        -----
        points : [num_samples, 2], np.float_
            Positions at which to interpolate the profile

        mode : int | `None`
            Mode of the profile to interpolate. If `None`.
            all modes are interpolated.
            Defaults to `None`.

        return_grads : bool
            If `True`, gradients and Hessians are computed.
            Defaults to `False`.

        Output
        ------
        values : [num_modes, num_samples] or [num_samples], np.float_
            Interpolated profile values at the sample positions

        grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
            Gradients of the interpolated profile values
            at the sample positions

        hessians : [num_modes, num_samples, 3, 3] or [num_samples,3,3], np.float_
            Hessians of the interpolated profile values
            at the sample positions
        """
        pass


class LagrangeProfileInterpolator(ProfileInterpolator):
    # pylint: disable=line-too-long
    r"""
    Class defining a :class:`~sionna.rt.ProfileInterpolator` using Lagrange polynomials

    The class instance is a callable that interpolates a discrete profile
    at arbitrary positions using two-dimensional 2nd-order Lagrange interpolation.

    A discrete profile :math:`P(y_i,z_j)\in\mathbb{R}` defined on
    a grid of points :math:`y_i,z_j` for :math:`i,j \in [1,2,3]` is
    interpolated to position :math:`y,z` as

    .. math::
        \begin{align}
            P(y,z) &= \sum_{i,j} P(y_i,z_j) \ell_{i,y}(y) \ell_{j,z}(z)
        \end{align}

    where :math:`\ell_{i,y}(y)`, :math:`\ell_{j,z}(z)` are the
    one-dimensional 2nd-order Lagrange polynomials, defined
    as

    .. math::
        \begin{align}
            \ell_{i,y}(y) &= \prod_{j \ne i} \frac{y-y_j}{y_i-y_j} \\
            \ell_{j,z}(z) &= \prod_{i \ne j} \frac{z-z_i}{z_j-z_i}.
        \end{align}

    Note that the formulation above assumes for simplicity only a 3x3 grid
    of points. However, the implementation finds for every
    position the closest 3x3 grid points of the discrete profile
    which are used for interpolation.

    In order to compute spatial gradients and Hessians, we extend the the profile
    with a dummy :math:`x` dimension, i.e., :math:`P(x,y,z)=P(y,z)`, such that

    .. math::
        \begin{align}
            \nabla P(x,y,z) &= \begin{bmatrix} 0, \frac{\partial P(x,y,z)}{\partial y}, \frac{\partial P(x,y,z)}{\partial z}  \end{bmatrix}^{\textsf{T}}\\
            H_P(x,y,z) &= \begin{bmatrix} 0 & 0                                                 & 0 \\
                                            0 & \frac{\partial^2 P(x,y,z)}{\partial y^2}          & \frac{\partial^2 P(x,y,z)}{\partial y \partial z} \\
                                            0 & \frac{\partial^2 P(x,y,z)}{\partial z \partial y} & \frac{\partial^2 P(x,y,z)}{\partial z^2}
                            \end{bmatrix}
        \end{align}.

    Parameters
    ----------
    discrete_profile : :class:`~sionna.rt.DiscreteProfile`
        Discrete profile to be interpolated

    Input
    -----
    points : [num_samples, 2], np.float_
        Positions at which to interpolate the profile

    mode : int | `None`
        Mode of the profile to interpolate. If `None`,
        all modes are interpolated.
        Defaults to `None`.

    return_grads : bool
        If `True`, gradients and Hessians are computed.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], np.float_
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated profile values
        at the sample positions

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
        Hessians of the interpolated profile values
        at the sample positions
    """

    @staticmethod
    def lagrange_polynomials(x, x_i, return_derivatives=True):
        # pylint: disable=line-too-long
        r"""
        Compute the 2nd-order Lagrange polynomials

        Optionally, the first- and second-order derivatives are returned.

        The 2nd-order Lagrange polynomials :math:`\ell_j(x)`, :math:`j=1,2,3`,
        for position :math:`x\in\mathbb{R}` are computed using three distinct
        support positions :math:`x_i` for :math:`i=1,2,3`:

        .. math::
            \begin{align}
                \ell_j(x) &= \prod_{\substack{1\leq i \leq 3 \\ i \ne j}} \frac{x-x_i}{x_j-x_i}.
            \end{align}

        Their first- and second-order derivatives are then respectively given as

        .. math::
            \begin{align}
                \ell'_j(x)  &= \left(\sum_{i \ne j} x-x_i\right) \left(\prod_{i \ne j} x_j-x_i\right)^{-1} \\
                \ell''_j(x) &= 2 \left(\prod_{i \ne j} x_j-x_i\right)^{-1}.
            \end{align}

        Input
        -----
        x : [batch_size], np.float_
            Sample positions

        x_i : [batch_size, 3], np.float_
            Support positions for every sample position

        return_derivatives : bool
            If `True`, also the first- and second-order derivatives
            of the Lagrange polynomials are returned.
            Defaults to `True`.

        Output
        ------
        l_i : [batch_size, 3], np.float_
            Lagrange polynomials for each sample position

        deriv_1st : [batch_size, 3], np.float_
            First-order derivatives for each sample position.
            Only returned if `return_derivatives` is `True`.

        deriv_2nd : [batch_size, 3], np.float_
            Second-order derivatives for each sample position.
            Only returned if `return_derivatives` is `True`.
        """

        # Compute products of differences of the sample and support points
        sample_diff = np.expand_dims(x, 1) - x_i
        sample_prod_0 = sample_diff[:, 1] * sample_diff[:, 2]
        sample_prod_1 = sample_diff[:, 0] * sample_diff[:, 2]
        sample_prod_2 = sample_diff[:, 0] * sample_diff[:, 1]
        sample_prods = np.stack([sample_prod_0, sample_prod_1, sample_prod_2], -1)

        # Compute products of differences of support points
        support_diffs = np.expand_dims(x_i, -1) - np.expand_dims(x_i, -2)
        support_diffs = np.where(support_diffs == 0, 1.0, support_diffs)
        support_prods = np.prod(support_diffs, axis=-1)

        # Compute Lagrange polynomials
        lagrange = sample_prods / support_prods

        if not return_derivatives:
            return lagrange
        else:
            # Compute sums of differences
            sample_sum_0 = sample_diff[:, 1] + sample_diff[:, 2]
            sample_sum_1 = sample_diff[:, 0] + sample_diff[:, 2]
            sample_sum_2 = sample_diff[:, 0] + sample_diff[:, 1]
            sample_sums = np.stack([sample_sum_0, sample_sum_1, sample_sum_2], -1)
            # Compute first-order derivatives
            deriv_1st = sample_sums / support_prods

            # Compute second-order derivatives
            deriv_2nd = 2. / support_prods

            return lagrange, deriv_1st, deriv_2nd

    def __call__(self, points, mode=None, return_grads=False):
        # pylint: disable=line-too-long
        r"""
        Interpolates a discrete profile at arbitrary position via
        2D 2nd-order Lagrange interpolation.

        A discrete profile :math:`P(y_i,z_j)\in\mathbb{R}` defined on
        a grid of points :math:`y_i,z_j` for :math:`i,j \in [1,2,3]` is
        interpolated to position :math:`y,z` as

        .. math::
            \begin{align}
                P(y,z) &= \sum_{i,j} P(y_i,z_j) \ell_{i,y}(y) \ell_{j,z}(z)
            \end{align}

        where :math:`\ell_{i,y}(y)`, :math:`\ell_{j,z}(z)` are the
        one-dimensional 2nd-order Lagrange polynomials, defined
        as

        .. math::
            \begin{align}
                \ell_{i,y}(y) &= \prod_{j \ne i} \frac{y-y_j}{y_i-y_j} \\
                \ell_{j,z}(z) &= \prod_{i \ne j} \frac{z-z_i}{z_j-z_i}.
            \end{align}

        In order to compute spatial gradients and Hessians, we extend the the profile
        with a dummy :math:`x` dimension, i.e., :math:`P(x,y,z)=P(y,z)`, such that

        .. math::
            \begin{align}
                \nabla P(x,y,z) &= \begin{bmatrix} 0, \frac{\partial P(x,y,z)}{\partial y}, \frac{\partial P(x,y,z)}{\partial z}  \end{bmatrix}^{\textsf{T}}\\
                H_P(x,y,z) &= \begin{bmatrix} 0 & 0                                                 & 0 \\
                                              0 & \frac{\partial^2 P(x,y,z)}{\partial y^2}          & \frac{\partial^2 P(x,y,z)}{\partial y \partial z} \\
                                              0 & \frac{\partial^2 P(x,y,z)}{\partial z \partial y} & \frac{\partial^2 P(x,y,z)}{\partial z^2}
                             \end{bmatrix}
            \end{align}.


        Input
        -----
        points : [num_samples, 2], np.float_
            Positions at which to interpolate the profile

        mode : int | `None`
            Mode of the profile to interpolate. If `None`,
            all modes are interpolated.
            Defaults to `None`.

        return_grads : bool
            If `True`, gradients and Hessians are computed.
            Defaults to `False`.

        Output
        ------
        values : [num_modes, num_samples] or [num_samples], np.float_
            Interpolated profile values at the sample positions

        grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
            Gradients of the interpolated profile values
            at the sample positions

        hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
            Hessians of the interpolated profile values
            at the sample positions
        """
        num_samples = np.shape(points)[0]
        # Compute absolute distances in y/z directions
        y_dist = np.abs(
            np.expand_dims(points[:, 0], axis=1)
            - np.expand_dims(self.cell_y_positions, axis=0)
        )
        z_dist = np.abs(
            np.expand_dims(points[:, 1], axis=1)
            - np.expand_dims(self.cell_z_positions, axis=0)
        )

        # Compute indices of three closest support points
        y_ind = np.sort(np.argmin(y_dist, axis=-1)[:, :3], -1)
        z_ind = np.sort(np.argmin(z_dist, axis=-1)[:, :3], -1)

        # Get support points in y and z dimensions
        y_i = np.take_along_axis(self.cell_y_positions, y_ind, axis=0)
        z_i = np.take_along_axis(self.cell_z_positions, z_ind, axis=0)

        # Compute indices of all support points
        support_ind = np.reshape(
            np.expand_dims(z_ind, 1) + np.expand_dims(y_ind, 2) * self.num_rows,
            [num_samples, -1],
        )

        # Compute support values for all modes
        vals = np.transpose(self.values, perm=[2, 1, 0])
        if mode is not None:
            # Filter relevant mode
            vals = np.expand_dims(vals[..., mode], -1)
        num_modes = np.shape(vals)[-1]
        vals = np.reshape(vals, [-1, num_modes])
        support_values = np.take_along_axis(vals, support_ind, axis=0)
        support_values = np.transpose(support_values, perm=[2, 0, 1])

        if not return_grads:
            # Compute Lagrange polynomials
            l_y = self.lagrange_polynomials(points[:, 0], y_i, False)
            l_z = self.lagrange_polynomials(points[:, 1], z_i, False)
            l_z_y = np.reshape(
                np.expand_dims(l_y, axis=-1) * np.expand_dims(l_z, axis=-2),
                [num_samples, -1],
            )

            # Compute interpolated values
            values = np.sum(support_values * l_z_y, axis=-1)
            return np.squeeze(values)

        # Compute Lagrange polynomials and derivatives
        l_y, d1_y, d2_y = self.lagrange_polynomials(points[:, 0], y_i, True)
        l_z, d1_z, d2_z = self.lagrange_polynomials(points[:, 1], z_i, True)
        l_z_y = np.reshape(
            np.expand_dims(l_y, axis=-1) * np.expand_dims(l_z, axis=-2),
            [num_samples, -1],
        )

        # Compute interpolated values
        values = np.sum(support_values * l_z_y, axis=-1)

        # Compute gradients
        l_z_d_y = np.reshape(
            np.expand_dims(d1_y, axis=-1) * np.expand_dims(l_z, axis=-2),
            [num_samples, -1],
        )

        d_values_dy = np.sum(support_values * l_z_d_y, axis=-1)

        l_d_z_y = np.reshape(
            np.expand_dims(l_y, axis=-1) * np.expand_dims(d1_z, axis=-2),
            [num_samples, -1],
        )
        d_values_dz = np.sum(support_values * l_d_z_y, axis=-1)

        grads = np.stack([np.zeros_like(d_values_dy), d_values_dy, d_values_dz], -1)

        # Compute Hessians
        # 1: Compute 2nd-order partial derivatives
        l_z_d2_y = np.reshape(
            np.expand_dims(d2_y, axis=-1) * np.expand_dims(l_z, axis=-2),
            [num_samples, -1],
        )
        d2_values_d2_y = np.sum(support_values * l_z_d2_y, axis=-1)

        l_d2_z_y = np.reshape(
            np.expand_dims(l_y, axis=-1) * np.expand_dims(d2_z, axis=-2),
            [num_samples, -1],
        )
        d2_values_d2_z = np.sum(support_values * l_d2_z_y, axis=-1)

        l_d_z_d_y = np.reshape(
            np.expand_dims(d1_y, axis=-1) * np.expand_dims(d1_z, axis=-2),
            [num_samples, -1],
        )
        d2_values_d_y_d_z = np.sum(support_values * l_d_z_d_y, axis=-1)

        # 2: Construct rows of the Hessians
        row_2 = np.stack(
            [np.zeros_like(d2_values_d2_y), d2_values_d2_y, d2_values_d_y_d_z], -1
        )

        row_3 = np.stack(
            [np.zeros_like(d2_values_d2_z), d2_values_d_y_d_z, d2_values_d2_z], -1
        )

        row_1 = np.zeros_like(row_2)

        # 3: Combine rows full Hessian matrices
        hessians = np.stack([row_1, row_2, row_3], axis=2)
        return (values, grads, hessians)


class DiscreteAmplitudeProfile(DiscreteProfile, AmplitudeProfile):
    # pylint: disable=line-too-long
    r"""Class defining a discrete amplitude profile of a RIS

    A discrete amplitude profile :math:`A_m` assigns to
    each of its units cells a possibly different amplitude value.
    Multiple reradiation modes can be obtained by super-positioning
    of profiles. The relative power of reradiation modes can
    be controlled via the reradiation coefficients :math:`p_m`.

    See :ref:`ris_primer` for more details.

    A class instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    cell_grid : :class:`~sionna.rt.CellGrid`
        Defines the physical structure of the RIS

    num_modes : int
        Number of reradiation modes.
        Defaults to 1.

    values : np.float_ or tf.Variable, [num_modes, num_rows, num_cols]
        Amplitude values for each reradiation mode
        and unit cell. `num_rows` and `num_cols` are defined by the
        `cell_grid`.
        Defaults to `None`.

    mode_powers : np.float_, [num_modes]
        Relative powers or reradition coefficients of reradiation modes.
        Defaults to `None`. In this case, all reradiation modes get
        an equal fraction of the total power.

    interpolator : :class:`~sionna.rt.ProfileInterpolator`
        Determines how the discrete values of the profile
        are interpolated to a continuous profile
        which is defined at any point on the RIS.
        Defaults to `None`. In this case, the
        :class:`~sionna.rt.LagrangeProfileInterpolator` will be used.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.

    Input
    -----
    points : np.float_, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], np.float_
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """

    def __init__(
        self,
        cell_grid,
        num_modes=1,
        values=None,
        mode_powers=None,
        interpolator=None,
        dtype=np.complex_,
    ):
        super().__init__(
            cell_grid=cell_grid,
            num_modes=num_modes,
            values=values,
            interpolator=interpolator,
            dtype=dtype,
        )

        if values is None:
            self.values = np.ones(self.shape, self._rdtype)

        if mode_powers is None:
            mode_powers = np.ones([self.num_modes], dtype=self._rdtype) / float(self.num_modes)
        self.mode_powers = mode_powers

    @property
    def mode_powers(self):
        return self._mode_powers

    @mode_powers.setter
    def mode_powers(self, v: np.ndarray):
        if isinstance(v, np.ndarray):
            if v.dtype != self._rdtype:
                msg = f"`mode_powers` must have dtype={self._rdtype}"
                raise TypeError(msg)
        else:
            v = np.ndarray(v, dtype=self._rdtype)

        if not v.shape == [self.num_modes]:
            msg = f"`mode_powers` must have shape [{self.num_modes}]"
            raise ValueError(msg)

        self._mode_powers = v


class DiscretePhaseProfile(DiscreteProfile, PhaseProfile):
    # pylint: disable=line-too-long
    r"""Class defining a discrete phase profile of a RIS

    A discrete phase profile :math:`\chi_m` assigns to
    each of its units cells a possibly different phase value.
    Multiple reradiation modes can be created by super-positioning
    of phase profiles.

    See :ref:`ris_primer` in the Primer on Electromagnetics for more details.

    A class instance is a callable that returns the profile values,
    gradients and Hessians at given points.

    Parameters
    ----------
    cell_grid : :class:`~sionna.rt.CellGrid`
        Defines the physical structure of the RIS

    num_modes : int
        Number of reradiation modes.
        Defaults to 1.

    values : np.float_ or tf.Variable, [num_modes, num_rows, num_cols]
        Phase values [rad] for each reradiation mode
        and unit cell. `num_rows` and `num_cols` are defined by the
        `cell_grid`.
        Defaults to `None`.

    interpolator : :class:`~sionna.rt.ProfileInterpolator`
        Determines how the discrete values of the profile
        are interpolated to a continuous profile
        which is defined at any point on the RIS.
        Defaults to `None`. In this case, the
        :class:`~sionna.rt.LagrangeProfileInterpolator` will be used.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.

    Input
    -----
    points : np.float_, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    values : [num_modes, num_samples] or [num_samples], np.float_
        Interpolated profile values at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
        Hessians of the interpolated profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """

    def __init__(
        self, cell_grid, num_modes=1, values=None, interpolator=None, dtype=np.complex_
    ):
        super().__init__(
            cell_grid=cell_grid,
            num_modes=num_modes,
            values=values,
            interpolator=interpolator,
            dtype=dtype,
        )

        if values is None:
            self.values = np.zeros(self.shape, self._rdtype)


class RIS(RadioDevice, SceneObject):
    # pylint: disable=line-too-long
    r"""
    Class defining a reconfigurable intelligent surface (RIS)

    A RIS consists of a planar arrangement of unit cells
    with :math:`\lambda/2` spacing.
    It's :class:`~sionna.rt.PhaseProfile` :math:`\chi_m` and
    :class:`~sionna.rt.AmplitudeProfile` :math:`A_m` can be
    configured after the RIS is instantiated. Both together
    define the spatial modulation coefficient :math:`\Gamma` which
    determines how the RIS reflects electro-magnetic waves.

    See :ref:`ris_primer` in the Primer on Electromagnetics for
    more details or have a look at the `tutorial notebook <https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_RIS.html>`_.

    An RIS instance is a callable that computes the spatial modulation coefficient
    and gradients/Hessians of the underlying phase profile for provided
    points on the RIS' surface.

    Parameters
    ----------
    name : str
        Name

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector

    num_rows : int
        Number of rows. Must at least be equal to three.

    num_cols : int
        Number of columns. Must at least be equal to three.

    num_modes : int
        Number of reradiation modes.
        Defaults to 1.

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0]. In this case, the normal vector of
        the RIS points towards the positive x-axis.

    velocity : [3], float
        Velocity vector [m/s]. Used for the computation of
        path-specific Doppler shifts.

    look_at : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.RIS` | :class:`~sionna.rt.Camera` | `None`
        A position or the instance of a :class:`~sionna.rt.Transmitter`,
        :class:`~sionna.rt.Receiver`, :class:`~sionna.rt.RIS`, or :class:`~sionna.rt.Camera` to look at.
        If set to `None`, then ``orientation`` is used to orientate the device.

    color : [3], float
        Defines the RGB (red, green, blue) ``color`` parameter for the device as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.
        Defaults to `[0.862,0.078,0.235]`.

    dtype : tf.complex
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.

    Input
    -----
    points : np.float_, [num_samples, 2]
        Tensor of 2D coordinates defining the points on the RIS at which
        the spatial modulation profile should be evaluated.
        Defaults to `None`. In this case, the values for all unit cells
        are returned.

    mode : int | `None`
        Reradiation mode to be considered.
        Defaults to `None`. In this case, the values for all modes
        are returned.

    return_grads : bool
        If `True`, also the first- and second-order derivatives are
        returned.
        Defaults to `False`.

    Output
    ------
    gamma : [num_modes, num_samples] or [num_samples], tf.complex
        Spatial modulation coefficient at the sample positions

    grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
        Gradients of the interpolated phase profile values
        at the sample positions. Only returned if `return_grads` is `True`.

    hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
        Hessians of the interpolated phase profile values
        at the sample positions. Only returned if `return_grads` is `True`.
    """

    def __init__(
        self,
        name,
        position,
        num_rows,
        num_cols,
        num_modes=1,
        orientation=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        look_at=None,
        color=(0.862, 0.078, 0.235),
        dtype=np.complex_,
    ):

        # Initialize the parent classes
        # RadioDevice and SceneObject inherit from Object
        # Python will initialize in the following order:
        # RadioDevice->SceneObject->Object
        super().__init__(
            name=name,
            position=position,
            orientation=orientation,
            look_at=look_at,
            radio_material=None,
            color=color,
            dtype=dtype,
        )

        # Set velocity vector
        self.velocity = np.asarray(velocity, np.float_)

        if num_rows < 3 or num_cols < 3:
            raise ValueError("num_rows and num_cols must be >= 3")

        # Set immutable properties
        self._num_modes = int(num_modes)
        self._cell_grid = CellGrid(num_rows, num_cols, self._dtype)

        # Init amplitude profile
        self.amplitude_profile = DiscreteAmplitudeProfile(
            self.cell_grid, num_modes=self.num_modes, dtype=self._dtype
        )

        # Init phase profile
        self.phase_profile = DiscretePhaseProfile(
            self.cell_grid, num_modes=self.num_modes, dtype=self._dtype
        )

    @property
    def cell_grid(self):
        r"""
        :class:`~sionna.rt.CellGrid` : Defines the physical
            structure of the RIS
        """
        return self._cell_grid

    @property
    def cell_positions(self):
        r"""
        [num_cells, 2], np.float_ : Cell positions in the
            local coordinate system (LCS) of the RIS, ordered
            from top-to-bottom left-to-right.
        """
        return self.cell_grid.cell_positions * self.spacing

    @property
    def cell_world_positions(self):
        r"""
        [num_cells, 3], np.float_ : Cell positions in the
            global coordinate system (GCS) of the RIS, ordered
            from top-to-bottom left-to-right.
        """
        x_coord = np.zeros([self.num_cells, 1], self._rdtype)
        pos = np.concatenate([x_coord, self.cell_positions], axis=-1)
        pos = rotate(pos, self.orientation)
        pos += np.expand_dims(self.position, 0)
        return pos

    @property
    def world_normal(self):
        r"""
        [3], np.float_ : Normal vector of the RIS in the
            global coordinate system (GCS)
        """
        n_hat = np.asarray([1., 0., 0.], self._rdtype)
        return rotate(n_hat, self.orientation)

    @property
    def num_rows(self):
        r"""
        int : Number of rows
        """
        return self.cell_grid.num_rows

    @property
    def num_cols(self):
        r"""
        int : Number of columns
        """
        return self.cell_grid.num_cols

    @property
    def num_cells(self):
        r"""
        int : Number of cells
        """
        return self.num_rows * self.num_cols

    @property
    def num_modes(self):
        r"""
        int : Number of reradiation modes
        """
        return self._num_modes

    @property
    def spacing(self):
        r"""
        np.float_: Element spacing [m] corresponding to
            half a wavelength
        """
        if hasattr(scene.Scene(), "wavelength"):
            wavelength = scene.Scene().wavelength
            return wavelength / 2.
        else:
            # Scene is not initialized
            return 0.5

    @property
    def size(self):
        """
        [2], np.float_ : Size of the RIS (width, height) [m]
        """
        return np.stack(
            [self.spacing * self.num_cols, self.spacing * self.num_rows], axis=0
        )

    @property
    def velocity(self):
        """
        [3], np.float_ : Get/set the velocity vector [m/s]
        """
        return self._velocity

    @velocity.setter
    def velocity(self, v: np.ndarray):
        if not np.shape(v) == 3:
            raise ValueError("`velocity` must have shape [3]")
        self._velocity = np.ndarray(v, float)

    @property
    def amplitude_profile(self):
        r"""
        :class:`~sionna.rt.AmplitudeProfile` : Set/get amplitude profile
        """
        return self._amplitude_profile

    @amplitude_profile.setter
    def amplitude_profile(self, v):
        if not isinstance(v, AmplitudeProfile):
            raise ValueError("Not a valid AmplitudeProfile")
        self._amplitude_profile = v

    @property
    def phase_profile(self):
        r"""
        :class:`~sionna.rt.PhaseProfile` : Set/get phase profile
        """
        return self._phase_profile

    @phase_profile.setter
    def phase_profile(self, v):
        if not isinstance(v, PhaseProfile):
            raise ValueError("Not a valid PhaseProfile")
        self._phase_profile = v

    def phase_gradient_reflector(self, sources, targets):
        # pylint: disable=line-too-long
        r"""
        Configures the RIS as ideal phase gradient reflector

        For an incoming direction :math:`\hat{\mathbf{k}}_i`
        and desired outgoing direction :math:`\hat{\mathbf{k}}_r`,
        the necessary phase gradient along the RIS with normal
        :math:`\hat{\mathbf{n}}` can be computed as
        (e.g., Eq.(12) [Vitucci24]_):

        .. math::
            \nabla\chi_m = -k_0\left( \mathbf{I}- \hat{\mathbf{n}}\hat{\mathbf{n}}^\textsf{T} \right) \left(\hat{\mathbf{k}}_i - \hat{\mathbf{k}}_r  \right).

        The phase profile is obtained by assigning zero phase to the first
        unit cell and evolving the other phases linearly according to the gradient
        across the entire RIS.

        Multiple reradiation modes can be configured.

        The amplitude profile is set to one everywhere with a uniform relative
        power allocation across modes.

        Input
        -----
        sources : np.float_, [3] or [num_modes, 3]
            Tensor defining for every reradiation mode
            a source from which the incoming wave originates.

        targets : np.float_, [3] or [num_modes, 3]
            Tensor defining for every reradiation mode
            a target towards which the incoming wave should be
            reflected.
        """
        # Convert inputs to tensors
        sources = np.ndarray(sources, self._rdtype)
        targets = np.ndarray(targets, self._rdtype)
        sources = sources[np.newaxis, np.newaxis]
        targets = targets[np.newaxis, np.newaxis]
        shape = [self.num_modes, 3]

        # Ensure the desired shape [num_modes, 3]
        for i, x in enumerate([sources, targets]):
            if not (np.shape(x) == shape).numpy().all():
                msg = (
                    f"Wrong shape of input {i+1}. " + f"Expected {shape}, got {x.shape}"
                )
                raise ValueError(msg)

        # Compute incoming and outgoing directions
        # [num_modes, 3]
        k_i, _ = normalize(self.position[np.newaxis] - sources)
        k_r, _ = normalize(targets - self.position[np.newaxis])

        # Tangent projection operator - Eq.(10)
        # [1, 3]
        normal = self.world_normal[np.newaxis]
        # [1, 3, 3]
        p = np.eye(3, dtype=self._rdtype) - outer(normal, normal)

        # Compute phase gradient - Eq.(12)
        # [num_modes, 3]
        grad = self.scene.wavenumber * (p @ (k_i - k_r))
        # Rotate phase gradient to LCS of the RIS and keep y/z components
        # [num_modes, 1, 1, 2]
        grad = rotate(grad, self.orientation, inverse=True)[:, 1:]
        grad = np.reshape(grad, [self.num_modes, 1, 1, 2])

        # Using the top-left cell as reference, compute the offsets
        # [1, num_rows, num_cols, 2]
        offsets = self.cell_positions - self.cell_positions[:1]
        offsets = np.reshape(offsets, [self.num_cols, self.num_rows, 2])
        offsets = np.transpose(offsets, perm=[1, 0, 2])
        offsets = np.expand_dims(offsets, 0)

        # Compute phase profile based on the constant gradient assumption
        # [num_modes, num_rows, num_cols]
        phases = np.sum(offsets * grad, axis=-1)
        self.phase_profile.values = phases

        # Set a neutral amplitude profile
        self.amplitude_profile.values = np.ones_like(phases)
        mode_powers = (
            np.ones([self.num_modes], dtype=self._rdtype)
            / float(self.num_modes)
        )
        self.amplitude_profile.mode_powers = mode_powers

    def focusing_lens(self, sources, targets):
        # pylint: disable=line-too-long
        r"""
        Configures the RIS as focusing lens

        The phase profile is configured in such a way that
        the fields of all rays add up coherently at a specific
        point. In other words, the phase profile undoes the
        distance-based phase shift of every ray connecting a
        source to a target via a specific unit cell.

        For a source and target at positions
        :math:`\mathbf{s}` and :math:`\mathbf{t}`, the phase
        :math:`\chi_m(\mathbf{x})` of a unit cell located at :math:`\mathbf{x}`
        is computed as (e.g., Sec. IV-2 [Degli-Esposti22]_)

        .. math::
            \chi_m(\mathbf{x}) = k_0 \left(\lVert\mathbf{s}-\mathbf{x}\rVert + \lVert\mathbf{s}-\mathbf{t}\rVert\right).

        Multiple reradiation modes can be configured.

        The amplitude profile is set to one everywhere with a uniform relative
        power allocation across modes.

        Input
        -----
        sources : np.float_, [3] or [num_modes, 3]
            Tensor defining for every reradiation mode
            a source from which the incoming wave originates.

        targets : np.float_, [3] or [num_modes, 3]
            Tensor defining for every reradiation mode
            a target towards which the incoming wave should be
            reflected.
        """
        # Convert inputs to tensors
        sources = np.asarray(sources, self._rdtype)
        targets = np.asarray(targets, self._rdtype)
        sources = sources[np.newaxis, np.newaxis]
        targets = targets[np.newaxis, np.newaxis]
        shape = [self.num_modes, 3]

        # Ensure the desired shape [num_modes, 3]
        for i, x in enumerate([sources, targets]):
            if not (np.shape(x) == shape).numpy().all():
                msg = (
                    f"Wrong shape of input {i+1}. " + f"Expected {shape}, got {x.shape}"
                )
                raise ValueError(msg)

        # Compute incoming and outgoing distances
        # [num_modes, num_cells]
        d_i = normalize(self.cell_world_positions[np.newaxis] - sources[:, np.newaxis])[
            1
        ]
        d_o = normalize(self.cell_world_positions[np.newaxis] - targets[:, np.newaxis])[
            1
        ]

        # Compute phases such that the total phase shifts for all cells
        # are equal
        phases = self.scene.wavenumber * (d_i + d_o)
        phases = np.reshape(phases, [self.num_modes, self.num_cols, self.num_rows])
        phases = np.transpose(phases, perm=[0, 2, 1])
        self.phase_profile.values = phases

        # Set a neutral amplitude profile
        self.amplitude_profile.values = np.ones_like(phases)
        mode_powers = (
            np.ones([self.num_modes], dtype=self._rdtype)
            / float(self.num_modes)
        )
        self.amplitude_profile.mode_powers = mode_powers

    def __call__(self, points=None, mode=None, return_grads=False):
        # pylint: disable=line-too-long
        r"""
        Computes the spatial modulation coefficient and gradients/Hessians of phase profile

        Input
        -----
        points : np.float_, [num_samples, 2]
            Tensor of 2D coordinates defining the points on the RIS at which
            the spatial modulation profile should be evaluated.
            Defaults to `None`. In this case, the values for all unit cells
            are returned.

        mode : int | `None`
            Reradiation mode to be considered.
            Defaults to `None`. In this case, the values for all modes
            are returned.

        return_grads : bool
            If `True`, also the first- and second-order derivatives are
            returned.
            Defaults to `False`.

        Output
        ------
        gamma : [num_modes, num_samples] or [num_samples], tf.complex
            Spatial modulation coefficient at the sample positions

        grads : [num_modes, num_samples, 3] or [num_samples, 3], np.float_
            Gradients of the interpolated phase profile values
            at the sample positions. Only returned if `return_grads` is `True`.

        hessians : [num_modes, num_samples, 3, 3] or [num_samples, 3, 3] , np.float_
            Hessians of the interpolated phase profile values
            at the sample positions. Only returned if `return_grads` is `True`.
        """
        # Get amplitudes
        a = self.amplitude_profile(points, mode)

        # Get mode powers
        p = self.amplitude_profile.mode_powers

        # Get phases and (optionally) phase gradients and Hessians
        if return_grads and points is not None:
            chi, grads, hessians = self.phase_profile(points, mode, True)
        else:
            chi = self.phase_profile(points, mode, False)

        # Compute spatial modulation coefficient
        gamma = a + 0.j
        chi = 1.j*chi
        p = np.sqrt(p) + 0.j
        gamma *= np.exp(chi)
        if mode is None:
            gamma *= np.reshape(p, [-1, 1])
        else:
            gamma *= p[mode]

        if return_grads and points is not None:
            return gamma, grads, hessians
        else:
            return gamma
