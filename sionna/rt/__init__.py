#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Ray tracing module of Sionna.
"""

###########################################
# Configuring Mitsuba variant
###########################################

import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

############################################
# Objects available to the user
############################################
# pylint: disable=wrong-import-position
from .scene import load_scene, Scene
from .camera import Camera
from .antenna import (
    Antenna,
    compute_gain,
    visualize,
    iso_pattern,
    dipole_pattern,
    hw_dipole_pattern,
    tr38901_pattern,
    polarization_model_1,
    polarization_model_2,
)
from .antenna_array import AntennaArray, PlanarArray
from .radio_material import RadioMaterial
from .ris import (
    AmplitudeProfile,
    DiscreteProfile,
    DiscreteAmplitudeProfile,
    DiscretePhaseProfile,
    CellGrid,
    PhaseProfile,
    RIS,
    ProfileInterpolator,
    LagrangeProfileInterpolator,
)
from .scene_object import SceneObject
from .scattering_pattern import (
    ScatteringPattern,
    LambertianPattern,
    DirectivePattern,
    BackscatteringPattern,
)
from .transmitter import Transmitter
from .receiver import Receiver
from .paths import Paths
from .coverage_map import CoverageMap
from .utils import (
    rotation_matrix,
    rotate,
    theta_phi_from_unit_vec,
    r_hat,
    theta_hat,
    phi_hat,
    cross,
    dot,
    outer,
    normalize,
    moller_trumbore,
    component_transform,
    reflection_coefficient,
    compute_field_unit_vectors,
    gen_orthogonal_vector,
    mi_to_np_ndarray,
    fibonacci_lattice,
    fibonacci_sphere,
    cot,
    sign,
    rot_mat_from_unit_vecs,
    sample_points_on_hemisphere,
)
