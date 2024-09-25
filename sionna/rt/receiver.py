#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class implementing a receiver
"""

import numpy as np
from .radio_device import RadioDevice


class Receiver(RadioDevice):
    # pylint: disable=line-too-long
    r"""
    Class defining a receiver

    Parameters
    ----------
    name : str
        Name

    position : [3], float
        Position :math:`(x,y,z)` as three-dimensional vector

    orientation : [3], float
        Orientation :math:`(\alpha, \beta, \gamma)` [rad] specified
        through three angles corresponding to a 3D rotation
        as defined in :eq:`rotation`.
        This parameter is ignored if ``look_at`` is not `None`.
        Defaults to [0,0,0].

    look_at : [3], float | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.RIS` | :class:`~sionna.rt.Camera` | None
        A position or the instance of a :class:`~sionna.rt.Transmitter`,
        :class:`~sionna.rt.Receiver`, :class:`~sionna.rt.RIS`, or :class:`~sionna.rt.Camera` to look at.
        If set to `None`, then ``orientation`` is used to orientate the device.

    color : [3], float
        Defines the RGB (red, green, blue) ``color`` parameter for the device as displayed in the previewer and renderer.
        Each RGB component must have a value within the range :math:`\in [0,1]`.
        Defaults to `[0.153, 0.682, 0.375]`.

    dtype : np.dtype
        Datatype to be used in internal calculations.
        Defaults to `np.complex_`.
    """

    def __init__(
        self,
        name,
        position,
        orientation=(0.0, 0.0, 0.0),
        look_at=None,
        color=(0.153, 0.682, 0.375),
        dtype=np.complex_,
    ):

        # Initialize the base class Object
        super().__init__(
            name=name,
            position=position,
            orientation=orientation,
            look_at=look_at,
            color=color,
            dtype=dtype,
        )
