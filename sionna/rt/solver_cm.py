#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Ray tracing algorithm that uses the image method to compute all pure reflection
paths.
"""

import mitsuba as mi
import drjit as dr
import numpy as np
from scipy.special import fresnel

from .utils import (
    matvec,
    dot,
    outer,
    phi_hat,
    theta_hat,
    theta_phi_from_unit_vec,
    normalize,
    rotation_matrix,
    mi_to_np_ndarray,
    compute_field_unit_vectors,
    reflection_coefficient,
    component_transform,
    fibonacci_lattice,
    r_hat,
    cross,
    cot,
    sign,
    sample_points_on_hemisphere,
    gen_basis_from_z,
    compute_spreading_factor,
    mitsuba_rectangle_to_world,
)
from .solver_base import SolverBase
from .coverage_map import CoverageMap
from .scattering_pattern import ScatteringPattern


class SolverCoverageMap(SolverBase):
    # pylint: disable=line-too-long
    r"""SolverCoverageMap(scene, solver=None, dtype=np.complex_)

    Generates a coverage map consisting of the squared amplitudes of the channel
    impulse response considering the LoS and reflection paths.

    The main inputs of the solver are:

    * The properties of the rectangle defining the coverage map, i.e., its
    position, scale, and orientation, and the resolution of the coverage map

    * The receiver orientation

    * A maximum depth, corresponding to the maximum number of reflections. A
    depth of zero corresponds to LoS only.

    Generation of a coverage map is carried-out for every transmitter in the
    scene. The antenna arrays of the transmitter and receiver are used.

    The generation of a coverage map consists in two steps:

    1. Shoot-and bounce ray tracing where rays are generated from the
    transmitters and the intersection with the rectangle defining the coverage
    map are recorded.
    Initial rays direction are arranged in a Fibonacci lattice on the unit
    sphere.

    2. The transfer matrices of every ray that intersect the coverage map are
    computed considering the materials of the objects that make the scene.
    The antenna patterns, synthetic phase shifts due to the array geometry, and
    combining and precoding vectors are then applied to obtain channel
    coefficients. The squared amplitude of the channel coefficients are then
    added to the value of the output corresponding to the cell of the coverage
    map within which the intersection between the ray and the coverage map
    occured.

    Note: Only triangle mesh are supported.

    Parameters
    -----------
    scene : :class:`~sionna.rt.Scene`
        Sionna RT scene

    solver : :class:`~sionna.rt.BaseSolver` | None
        Another solver from which to re-use some structures to avoid useless
        compute and memory use

    dtype : np.dtype
        Datatype for all computations, inputs, and outputs.
        Defaults to `np.complex_`.

    Input
    ------
    max_depth : int
        Maximum depth (i.e., number of bounces) allowed for tracing the
        paths

    rx_orientation : (3,), np.float_
        Orientation of the receiver.
        This is used to compute the antenna response and antenna pattern
        for an imaginary receiver located on the coverage map.

    cm_center : (3,), np.float_
        Center of the coverage map

    cm_orientation : (3,), np.float_
        Orientation of the coverage map

    cm_size : (2,), np.float_
        Scale of the coverage map.
        The width of the map (in the local X direction) is scale[0]
        and its map (in the local Y direction) scale[1].

    cm_cell_size : (2,), np.float_
        Resolution of the coverage map, i.e., width
        (in the local X direction) and height (in the local Y direction) in
        meters of a cell of the coverage map

    combining_vec : (num_rx,), np.complex_ | None
        Combining vector.
        This is used to combine the signal from the receive antennas for
        an imaginary receiver located on the coverage map.
        If set to `None`, then no combining is applied, and
        the energy received by all antennas is summed.

    precoding_vec : (num_tx,), np.complex_
        Precoding vectors of the transmitters

    num_samples : int
        Number of rays initially shooted from the transmitters.
        This number is shared by all transmitters, i.e.,
        ``num_samples/num_tx`` are shooted for each transmitter.

    los : bool
        If set to `True`, then the LoS paths are computed.

    reflection : bool
        If set to `True`, then the reflected paths are computed.

    diffraction : bool
        If set to `True`, then the diffracted paths are computed.

    scattering : bool
        if set to `True`, then the scattered paths are computed.

    ris : bool
        If set to `True`, then paths involving RIS are computed.

    edge_diffraction : bool
        If set to `False`, only diffraction on wedges, i.e., edges that
        connect two primitives, is considered.

    Output
    -------
    :cm : :class:`~sionna.rt.CoverageMap`
        The coverage maps
    """

    DISCARD_THRES = 1e-15  # -150 dB

    def __call__(
        self,
        max_depth,
        rx_orientation,
        cm_center,
        cm_orientation,
        cm_size,
        cm_cell_size,
        combining_vec,
        precoding_vec,
        num_samples,
        los,
        reflection,
        diffraction,
        scattering,
        ris,
        edge_diffraction,
    ):

        # If reflection and scattering are disabled, no need for a max_depth
        # higher than 1.
        # This clipping can save some compute for the shoot-and-bounce
        if (not reflection) and (not scattering):
            max_depth = np.minimum(max_depth, 1)

        # Transmitters positions and orientations
        # sources_positions : (num_tx, 3)
        # sources_orientations : (num_tx, 3)
        sources_positions = []
        sources_orientations = []
        for tx in self._scene.transmitters.values():
            sources_positions.append(tx.position)
            sources_orientations.append(tx.orientation)
        sources_positions = np.stack(sources_positions, axis=0)
        sources_orientations = np.stack(sources_orientations, axis=0)

        # EM properties of the materials
        # Returns: relative_permittivities, denoted by `etas`
        # scattering_coefficients, xpd_coefficients,
        # alpha_r, alpha_i and lambda_
        object_properties = self._build_scene_object_properties_tensors()
        etas = object_properties[0]
        scattering_coefficient = object_properties[1]
        xpd_coefficient = object_properties[2]
        alpha_r = object_properties[3]
        alpha_i = object_properties[4]
        lambda_ = object_properties[5]

        # Measurement plane defining the coverage map
        # meas_plane : mi.Shape
        #     Mitsuba rectangle defining the measurement plane
        meas_plane = self._build_mi_measurement_plane(
            cm_center, cm_orientation, cm_size
        )

        # Builds the Mitsuba scene with RIS for
        # testing intersections with RIS
        mi_ris_scene = self._build_mi_ris_objects()

        ####################################################
        # Shooting-and-bouncing
        # Computes the coverage map for LoS, reflection,
        # and scattering.
        # Also returns the primitives found in LoS of the
        # transmitters to shoot diffracted rays.
        ####################################################

        cm, los_primitives = self._shoot_and_bounce(
            meas_plane,
            mi_ris_scene,
            rx_orientation,
            sources_positions,
            sources_orientations,
            max_depth,
            num_samples,
            combining_vec,
            precoding_vec,
            cm_center,
            cm_orientation,
            cm_size,
            cm_cell_size,
            los,
            reflection,
            diffraction,
            scattering,
            ris,
            etas,
            scattering_coefficient,
            xpd_coefficient,
            alpha_r,
            alpha_i,
            lambda_,
        )

        # ############################################
        # # Diffracted
        # ############################################

        if los_primitives is not None:

            cm_diff = self._diff_samples_2_coverage_map(
                los_primitives,
                edge_diffraction,
                num_samples,
                sources_positions,
                meas_plane,
                cm_center,
                cm_orientation,
                cm_size,
                cm_cell_size,
                sources_orientations,
                rx_orientation,
                combining_vec,
                precoding_vec,
                etas,
                scattering_coefficient,
            )

            cm = cm + cm_diff

        # ############################################
        # # Combine the coverage maps.
        # # Coverage maps are combined non-coherently
        # ############################################
        cm = CoverageMap(
            cm_center,
            cm_orientation,
            cm_size,
            cm_cell_size,
            cm,
            scene=self._scene,
            dtype=self._dtype,
        )
        return cm

    ##################################################################
    # Internal methods
    ##################################################################

    def _build_mi_measurement_plane(self, cm_center, cm_orientation, cm_size):
        r"""
        Builds the Mitsuba rectangle defining the measurement plane
        corresponding to the coverage map.

        Input
        ------
        cm_center : (3,), np.float_
            Center of the rectangle

        cm_orientation : (3,), np.float_
            Orientation of the rectangle

        cm_size : (2,), np.float_
            Scale of the rectangle.
            The width of the rectangle (in the local X direction) is scale[0]
            and its height (in the local Y direction) scale[1].

        Output
        ------
        mi_meas_plane : mi.Shape
            Mitsuba rectangle defining the measurement plane
        """
        # Rectangle defining the coverage map
        mi_meas_plane = mi.load_dict(
            {
                "type": "rectangle",
                "to_world": mitsuba_rectangle_to_world(
                    cm_center, cm_orientation, cm_size
                ),
            }
        )

        return mi_meas_plane

    def _mp_hit_point_2_cell_ind(
        self, rot_gcs_2_mp, cm_center, cm_size, cm_cell_size, num_cells, hit_point
    ):
        r"""
        Computes the indices of the cells to which points ``hit_point`` on the
        measurement plane belongs.

        Input
        ------
        rot_gcs_2_mp : (3, 3), np.float_
            Rotation matrix for going from the measurement plane LCS to the GCS

        cm_center : (3,), np.float_
            Center of the coverage map

        cm_size : (2,), np.float_
            Size of the coverage map

        cm_cell_size : (2,), np.float_
            Size of of the cells of the ceverage map

        num_cells : (2,), np.int_
            Number of cells in the coverage map

        hit_point : (...,3)
            Intersection points

        Output
        -------
        cell_ind : (..., 2), np.int_
            Indices of the cells
        """

        # Expand for broadcasting
        # (..., 3, 3)
        num_extra_dims = max(0, len(hit_point.shape) - 1)
        rot_gcs_2_mp = rot_gcs_2_mp.reshape(((1,) * num_extra_dims + (3, 3)))
        # (..., 3)
        num_extra_dims = max(0, len(hit_point.shape))
        cm_center = cm_center.reshape(((1,) * num_extra_dims + (3,)))

        # Coverage map cells' indices
        # Coordinates of the hit point in the coverage map LCS
        # (..., 3)
        hit_point = matvec(rot_gcs_2_mp, hit_point - cm_center)

        # In the local coordinate system of the coverage map, z should be 0
        # as the coverage map is in XY

        # x
        # (...)
        cell_x = hit_point[..., 0] + cm_size[0] * 0.5
        cell_x = np.floor(cell_x / cm_cell_size[0]).astype(np.int_)
        cell_x = np.where(cell_x < num_cells[0], cell_x, num_cells[0])
        cell_x = np.where(cell_x >= 0, cell_x, num_cells[0])

        # y
        # (...)
        cell_y = hit_point[..., 1] + cm_size[1] * 0.5
        cell_y = np.floor(cell_y / cm_cell_size[1]).astype(np.int_)
        cell_y = np.where(cell_y < num_cells[1], cell_y, num_cells[1])
        cell_y = np.where(cell_y >= 0, cell_y, num_cells[1])

        # (..., 2)
        cell_ind = np.stack([cell_y, cell_x], axis=-1)

        return cell_ind

    def _compute_antenna_patterns(self, rot_mat, patterns, k):
        r"""
        Evaluates the antenna ``patterns`` of a radio device with oriented
        following ``orientation``, and for a incident field direction
        ``k``.

        Input
        ------
        rot_mat : (..., 3, 3), np.float_
            Rotation matrix built from the orientation of the radio device

        patterns : (f(theta, phi)], list of callable
            List of antenna patterns

        k : (..., 3), np.float_
            Direction of departure/arrival in the GCS.
            Must point away from the radio device

        Output
        -------
        fields_hat : (..., num_patterns, 2), np.complex_
            Antenna fields theta_hat and phi_hat components in the GCS

        theta_hat : (..., 3), np.float_
            Theta hat direction in the GCS

        phi_hat : (..., 3), np.float_
            Phi hat direction in the GCS

        """

        # (..., 3, 3)
        num_extra_dims = max(0, k.ndim + 1 - rot_mat.ndim)
        rot_mat = rot_mat.reshape(((1,) * num_extra_dims + rot_mat.shape))

        # (...)
        theta, phi = theta_phi_from_unit_vec(k)

        # Normalized direction vector in the LCS of the radio device
        # (..., 3)
        k_prime = matvec(rot_mat, k, True)

        # Angles of departure in the local coordinate system of the
        # radio device
        # (...)
        theta_prime, phi_prime = theta_phi_from_unit_vec(k_prime)

        # Spherical global frame vectors
        # (..., 3)
        theta_hat_ = theta_hat(theta, phi)
        phi_hat_ = phi_hat(phi)

        # Spherical local frame vectors
        # (..., 3)
        theta_hat_prime = theta_hat(theta_prime, phi_prime)
        phi_hat_prime = phi_hat(phi_prime)

        # Rotate the LCS according to the radio device orientation
        # (..., 3)
        theta_hat_prime = matvec(rot_mat, theta_hat_prime)
        phi_hat_prime = matvec(rot_mat, phi_hat_prime)

        # Rotation matrix for going from the spherical radio device LCS to the
        # spherical GCS
        # (..., 2, 2)
        lcs2gcs = component_transform(
            theta_hat_prime, phi_hat_prime, theta_hat_, phi_hat_  # LCS
        )  # GCS
        lcs2gcs = lcs2gcs + 0.0j

        # Compute the fields in the LCS
        fields_hat = []
        for pattern in patterns:
            # (..., 2)
            field_ = np.stack(pattern(theta_prime, phi_prime), axis=-1)
            fields_hat.append(field_)

        # Stacking the patterns, corresponding to different polarization
        # directions, as an additional dimension
        # (..., num_patterns, 2)
        fields_hat = np.stack(fields_hat, axis=-2)

        # Fields in the GCS
        # (..., 1, 2, 2)
        lcs2gcs = np.expand_dims(lcs2gcs, axis=-3)
        # (..., num_patterns, 2)
        fields_hat = matvec(lcs2gcs, fields_hat)

        return fields_hat, theta_hat_, phi_hat_

    def _apply_synthetic_array(self, tx_rot_mat, rx_rot_mat, k_rx, k_tx, a):
        # pylint: disable=line-too-long
        r"""
        Synthetically apply transmitter and receiver arrays to the channel
        coefficients ``a``

        Input
        ------
        tx_rot_mat : (..., 3, 3), np.float_
            Rotation matrix built from the orientation of the transmitters

        rx_rot_mat : (3, 3), np.float_
            Rotation matrix built from the orientation of the receivers

        k_rx : (..., 3), np.float_
            Directions of arrivals of the rays

        k_tx : (..., 3), np.float_
            Directions of departure of the rays

        a : (..., num_rx_patterns, num_tx_patterns), np.complex_
            Channel coefficients

        Output
        -------
        a : (..., num_rx_ant, num_tx_ant), np.complex_
            Channel coefficients with the antenna array applied
        """

        two_pi = 2.0 * np.pi

        # Rotated position of the TX antenna elements
        # (..., tx_array_size, 3)
        num_extra_dims = max(0, tx_rot_mat.ndim - self._scene.tx_array.positions.ndim)
        tx_rel_ant_pos = self._scene.tx_array.positions.reshape(
            (1,) * num_extra_dims + self._scene.tx_array.positions.shape
        )

        # (..., 1, 3, 3)
        tx_rot_mat_ = np.expand_dims(tx_rot_mat, axis=-3)
        # (..., tx_array_size, 3)
        tx_rel_ant_pos = matvec(tx_rot_mat_, tx_rel_ant_pos)

        # Rotated position of the RX antenna elements
        # (1, rx_array_size, 3)
        rx_rel_ant_pos = self._scene.rx_array.positions
        # (1, 3, 3)
        rx_rot_mat = np.expand_dims(rx_rot_mat, axis=0)
        # (rx_array_size, 3)
        rx_rel_ant_pos = matvec(rx_rot_mat, rx_rel_ant_pos)
        # (..., rx_array_size, 3)
        num_extra_dims = max(0, tx_rel_ant_pos.ndim - rx_rel_ant_pos.ndim)
        rx_rel_ant_pos = rx_rel_ant_pos.reshape(
            (1,) * num_extra_dims + rx_rel_ant_pos.shape
        )

        # Expand dims for broadcasting with antennas
        # (..., 1, 3)
        k_rx = np.expand_dims(k_rx, axis=-2)
        k_tx = np.expand_dims(k_tx, axis=-2)
        # Compute the synthetic phase shifts due to the antenna array
        # Transmitter side
        # (..., tx_array_size)
        tx_phase_shifts = dot(tx_rel_ant_pos, k_tx)
        # Receiver side
        # (..., rx_array_size)
        rx_phase_shifts = dot(rx_rel_ant_pos, k_rx)
        # Total phase shift
        # (..., rx_array_size, 1)
        rx_phase_shifts = np.expand_dims(rx_phase_shifts, axis=-1)
        # (..., 1, tx_array_size)
        tx_phase_shifts = np.expand_dims(tx_phase_shifts, axis=-2)
        # (..., rx_array_size, tx_array_size)
        phase_shifts = rx_phase_shifts + tx_phase_shifts
        phase_shifts = two_pi * phase_shifts / self._scene.wavelength
        # Apply the phase shifts
        # (..., 1, rx_array_size, 1, tx_array_size)
        phase_shifts = np.expand_dims(phase_shifts, axis=-2)
        phase_shifts = np.expand_dims(phase_shifts, axis=-4)
        # (..., num_rx_patterns, 1, num_tx_patterns, 1)
        a = np.expand_dims(a, axis=-1)
        a = np.expand_dims(a, axis=-3)
        # (..., num_rx_patterns, rx_array_size, num_tx_patterns, tx_array_size)
        a = a * np.exp(phase_shifts * 1.0j)
        # Reshape to merge antenna patterns and array
        # (...,
        #   num_rx=num_rx_patterns*rx_array_size,
        #   num_tx=num_tx_patterns*tx_array_size )
        a = a.reshape(
            a.shape[:-4] + (a.shape[-4] * a.shape[-3], a.shape[-2] * a.shape[-1])
        )

        return a

    def _update_coverage_map(
        self,
        cm_center,
        cm_size,
        cm_cell_size,
        num_cells,
        rot_gcs_2_mp,
        cm_normal,
        tx_rot_mat,
        rx_rot_mat,
        precoding_vec,
        combining_vec,
        samples_tx_indices,
        e_field,
        field_es,
        field_ep,
        mp_hit_point,
        hit_mp,
        k_tx,
        previous_int_point,
        cm,
        radii_curv,
        angular_opening,
    ):
        r"""
        Updates the coverage map with the power of the paths that hit it.

        Input
        ------
        cm_center : (3,), np.float_
            Center of the coverage map

        cm_size : (2,), np.float_
            Scale of the coverage map.
            The width of the map (in the local X direction) is ``cm_size[0]``
            and its map (in the local Y direction) ``cm_size[1]``.

        cm_cell_size : (2,), np.float_
            Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            meters of a cell of the coverage map

        num_cells : (2,), np.int_
            Number of cells in the coverage map

        rot_gcs_2_mp : (3, 3), np.float_
            Rotation matrix for going from the measurement plane LCS to the GCS

        cm_normal : (3,), np.float_
            Normal to the measurement plane

        tx_rot_mat : (num_tx, 3, 3), np.float_
            Rotation matrix built from the orientation of the transmitters

        rx_rot_mat : (3, 3), np.float_
            Rotation matrix built from the orientation of the receivers

        precoding_vec : (num_tx,), np.complex_
            Vector used for transmit-precoding

        combining_vec : (num_rx_ant,), np.complex_ | None
            Vector used for receive-combing.
            If set to `None`, then no combining is applied, and
            the energy received by all antennas is summed.

        samples_tx_indices : (num_samples,), np.int_
            Transmitter indices that correspond to every sample, i.e., from
            which the ray was shot.

        e_field : (num_samples, num_tx_patterns, 2), np.float_
            Incoming electric field. These are the e_s and e_p components.
            The e_s and e_p directions are given thereafter.

        field_es : (num_samples, 3), np.float_
            S direction for the incident field

        field_ep : (num_samples, 3), np.float_
            P direction for the incident field

        mp_hit_point : (num_samples, 3), np.float_
            Positions of the hit points with the measurement plane.

        hit_mp : (num_samples,), np.bool_
            Set to `True` for samples that hit the measurement plane

        k_tx : (num_samples, 3,), np.float_
            Direction of departure from the transmitters

        previous_int_point : (num_samples, 3), np.float_
            Position of the previous interaction with the scene

        cm : (num_tx, num_cells_y+1, num_cells_x+1), np.float_
            Coverage map

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature

        angular_opening : (num_active_samples,), np.float_
            Angular opening of the ray tube

        Output
        -------
        cm : (num_tx, num_cells_y+1, num_cells_x+1), np.float_
            Updated coverage map
        """

        # Extract the samples that hit the coverage map.
        # This is to avoid computing the channel coefficients for all the
        # samples.
        # Indices of the samples that hit the coverage map
        # (num_hits,)
        hit_mp_ind = np.argwhere(hit_mp)[:, 0]
        # Indices of the transmitters corresponding to the rays that hit
        # (num_hits,)
        hit_mp_tx_ind = samples_tx_indices[hit_mp_ind]
        # the coverage map
        # (num_hits, 3)
        mp_hit_point = mp_hit_point[hit_mp_ind]
        # (num_hits, 3)
        previous_int_point = previous_int_point[hit_mp_ind]
        # (num_hits, 3)
        k_tx = k_tx[hit_mp_ind]
        # (num_hits, 3)
        precoding_vec = precoding_vec[hit_mp_tx_ind]
        # (num_hits, 3, 3)
        tx_rot_mat = tx_rot_mat[hit_mp_tx_ind]
        # (num_hits, num_tx_patterns, 2)
        e_field = e_field[hit_mp_ind]
        # (num_hits, 3)
        field_es = field_es[hit_mp_ind]
        # (num_hits, 3)
        field_ep = field_ep[hit_mp_ind]
        # (num_hits, 2)
        radii_curv = radii_curv[hit_mp_ind]
        # (num_hits,)
        angular_opening = angular_opening[hit_mp_ind]

        # Cell indices
        # (num_hits, 2)
        hit_cells = self._mp_hit_point_2_cell_ind(
            rot_gcs_2_mp, cm_center, cm_size, cm_cell_size, num_cells, mp_hit_point
        )
        # Receive direction
        # k_rx : (num_hits, 3)
        # length : (num_hits,)
        k_rx, length = normalize(mp_hit_point - previous_int_point)

        # Apply spreading factor
        # (num_active_samples,)
        sf = compute_spreading_factor(radii_curv[:, 0], radii_curv[:, 1], length)
        # (num_active_samples, 1, 1)
        num_extra_dims = max(0, e_field.ndim - sf.ndim)
        sf = sf.reshape(sf.shape + (1,) * num_extra_dims)
        sf = sf + 0.0j
        # (num_active_samples, num_tx_patterns, 2)
        e_field *= sf

        # Compute the receive field in the GCS
        # rx_field : (num_hits, num_rx_patterns, 2)
        # rx_es_hat, rx_ep_hat : (num_hits, 3)
        rx_field, rx_es_hat, rx_ep_hat = self._compute_antenna_patterns(
            rx_rot_mat, self._scene.rx_array.antenna.patterns, -k_rx
        )
        # Move the incident field to the receiver basis
        # Change of basis of the field
        # (num_hits, 2, 2)
        to_rx_mat = component_transform(field_es, field_ep, rx_es_hat, rx_ep_hat)
        # (num_hits, 1, 2, 2)
        to_rx_mat = np.expand_dims(to_rx_mat, axis=1)
        to_rx_mat = to_rx_mat + 0.0j
        # (num_hits, num_tx_patterns, 2)
        e_field = matvec(to_rx_mat, e_field)
        # Apply the receiver antenna field to compute the channel coefficient
        # (num_hits num_rx_patterns, 1, 2)
        rx_field = np.expand_dims(rx_field, axis=2)
        # (num_hits, 1, num_tx_patterns, 2)
        e_field = np.expand_dims(e_field, axis=1)

        # (num_hits, num_rx_patterns, num_tx_patterns)
        a = np.sum(np.conj(rx_field) * e_field, axis=-1)

        # Apply synthetic array
        # (num_hits, num_rx_ant, num_tx_ant)
        a = self._apply_synthetic_array(tx_rot_mat, rx_rot_mat, k_rx, k_tx, a)

        # Apply precoding
        # (num_hits, 1, num_tx_ant)
        precoding_vec = np.expand_dims(precoding_vec, 1)
        # (num_hits, num_rx_ant)
        a = np.sum(a * precoding_vec, axis=-1)
        # Apply combining
        # If no combining vector is provided, then sum the energy received by
        # the antennas
        if combining_vec is None:
            # (num_hits,)
            a = np.sum(np.abs(a) ** 2, axis=-1)
        else:
            # (1, num_rx_ant)
            combining_vec = np.expand_dims(combining_vec, 0)
            # (num_hits,)
            a = np.sum(np.conj(combining_vec) * a, axis=-1)
            # Compute the amplitude of the path
            # (num_hits,)
            a = np.abs(a) ** 2

        # Add the rays contribution to the coverage map
        # We just divide by cos(aoa) instead of dividing by the square distance
        # to apply the propagation loss, to then multiply by the square distance
        # over cos(aoa) to compute the ray weight.
        # Ray weighting
        # Cosine of the angle of arrival with respect to the normal of
        # the plan
        # (num_hits,)
        cos_aoa = np.abs(dot(k_rx, cm_normal, clip=True))

        # Radii of curvature at the interaction point with the measurement plane
        # (num_hits, 2)
        radii_curv += np.expand_dims(length, axis=1)
        # (num_hits,)
        ray_weights = np.where(
            cos_aoa == 0.0, 0.0, radii_curv[:, 0] * radii_curv[:, 1] / cos_aoa
        )
        ray_weights *= angular_opening
        # Add the contribution to the coverage map
        # (num_hits, 3)
        hit_cells = np.concatenate(
            [np.expand_dims(hit_mp_tx_ind, axis=-1), hit_cells], axis=-1
        )
        # (num_tx, num_cells_y+1, num_cells_x+1)
        cm[hit_cells] += ray_weights * a

        return cm

    def _compute_reflected_field(
        self,
        normals,
        etas,
        scattering_coefficient,
        k_i,
        e_field,
        field_es,
        field_ep,
        scattering,
        length,
        radii_curv,
        dirs_curv,
    ):
        r"""
        Computes the reflected field at the intersections.

        Input
        ------
        normals : (num_active_samples, 3), np.float_
            Normals to the intersected primitives

        etas : (num_active_samples,), np.complex_
            Relative permittivities of the intersected primitives

        scattering_coefficient : (num_active_samples,), np.float_
            Scattering coefficients of the intersected primitives

        k_i : (num_active_samples, 3), np.float_
            Direction of arrival of the ray

        e_field : (num_active_samples, num_tx_patterns, 2), np.complex_
            S and P components of the incident field

        field_es : (num_active_samples, 3), np.float_
            Direction of the S component of the incident field

        field_ep : (num_active_samples, 3), np.float_
            Direction of the P component of the incident field

        scattering : bool
            Set to `True` if scattering is enabled

        length : (num_active_samples,), np.float_
            Length of the last path segment

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature

        dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal direction of curvature

        Output
        -------
        e_field : (num_active_samples, num_tx_patterns, 2), np.complex_
            S and P components of the reflected field

        field_es : (num_active_samples, 3), np.float_
            Direction of the S component of the reflected field

        field_ep : (num_active_samples, 3), np.float_
            Direction of the P component of the reflected field

        k_r : (num_active_samples, 3), np.float_
            Direction of the reflected ray

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature of the reflected ray tube

        dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal direction of curvature of the reflected ray tube
        """

        # (num_active_samples, 3)
        k_r = k_i - 2.0 * dot(k_i, normals, keepdim=True, clip=True) * normals

        # S/P direction for the incident/reflected field
        # (num_active_samples, 3)
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s, e_i_p, e_r_s, e_r_p = compute_field_unit_vectors(
            k_i, k_r, normals, SolverBase.EPSILON
        )

        # Move to the incident S/P component
        # (num_active_samples, 2, 2)
        to_incident = component_transform(field_es, field_ep, e_i_s, e_i_p)
        # (num_active_samples, 1, 2, 2)
        to_incident = np.expand_dims(to_incident, axis=1)
        to_incident = to_incident + 0.0j
        # (num_active_samples, num_tx_patterns, 2)
        e_field = matvec(to_incident, e_field)

        # Compute the reflection coefficients
        # (num_active_samples,)
        cos_theta = -dot(k_i, normals, clip=True)

        # (num_active_samples,)
        r_s, r_p = reflection_coefficient(etas, cos_theta)

        # If scattering is enabled, then the rays are randomly
        # allocated to reflection or scattering by sampling according to the
        # scattering coefficient. An oversampling factor is applied to keep
        # differentiability with respect to the scattering coefficient.
        # This oversampling factor is the ratio between the reduction factor
        # and the (non-differientiable) probability with which a
        # reflection phenomena is selected. In our case, this probability is the
        # reduction factor.
        # If scattering is disabled, all samples are allocated to reflection to
        # maximize sample-efficiency. However, this requires correcting the
        # contribution of the reflected rays by applying the reduction factor.
        # (num_active_samples,)
        reduction_factor = np.sqrt(1.0 - scattering_coefficient**2)
        reduction_factor = reduction_factor + 0.0j
        if scattering:
            # (num_active_samples,)
            ovs_factor = np.where(reduction_factor == 0.0, 0.0, 1.0)
            r_s *= ovs_factor
            r_p *= ovs_factor
        else:
            # (num_active_samples,)
            r_s *= reduction_factor
            r_p *= reduction_factor

        # Apply the reflection coefficients
        # (num_active_samples, 2)
        r = np.stack([r_s, r_p], -1)
        # (num_active_samples, 1, 2)
        r = np.expand_dims(r, axis=-2)
        # (num_active_samples, num_tx_patterns, 2)
        e_field *= r

        # Update S/P directions
        # (num_active_samples, 3)
        field_es = e_r_s
        field_ep = e_r_p

        # Compute and apply the spreading factor
        # (num_active_samples,)
        sf = compute_spreading_factor(radii_curv[:, 0], radii_curv[:, 1], length)
        # (num_active_samples, 1, 1)
        num_extra_dims = max(0, e_field.ndim - sf.ndim)
        sf = sf.reshape(sf.shape + (1,) * num_extra_dims)
        sf = sf + 0.0j
        # (num_active_samples, num_tx_patterns, 2)
        e_field *= sf

        # Update principal radii of curvature
        # Radii of curvature at intersection point
        # (num_reflected_samples, 2)
        radii_curv += np.expand_dims(length, axis=1)
        # Radii of curvature of the reflected field
        # (num_reflected_samples, 2)
        inv_radii_curv = np.where(radii_curv == 0.0, 0.0, 1.0 / radii_curv)
        # (num_reflected_samples,)
        inv_radii_curv_sum = inv_radii_curv[:, 0] + inv_radii_curv[:, 1]
        # (num_reflected_samples,)
        inv_radii_curv_dif = np.abs(inv_radii_curv[:, 0] - inv_radii_curv[:, 1])
        # (num_reflected_samples, 2)
        inv_new_radii_curv = np.stack(
            [
                0.5 * (inv_radii_curv_sum + inv_radii_curv_dif),
                0.5 * (inv_radii_curv_sum - inv_radii_curv_dif),
            ],
            axis=1,
        )
        # (num_reflected_samples, 2)
        new_radii_curv = np.where(
            inv_new_radii_curv == 0.0, 0.0, 1.0 / inv_new_radii_curv
        )

        # Update the principal direction of curvature
        # (num_reflected_samples, 3)
        new_dir_curv_1 = (
            dirs_curv[:, 0]
            - 2.0 * dot(dirs_curv[:, 0], normals, keepdim=True, clip=True) * normals
        )
        # (num_reflected_samples, 3)
        new_dir_curv_2 = -cross(k_r, new_dir_curv_1)
        # (num_reflected_samples, 2, 3)
        new_dirs_curv = np.stack([new_dir_curv_1, new_dir_curv_2], axis=1)

        return e_field, field_es, field_ep, k_r, new_radii_curv, new_dirs_curv

    def _compute_scattered_field(
        self,
        int_point,
        objects,
        normals,
        etas,
        scattering_coefficient,
        xpd_coefficient,
        alpha_r,
        alpha_i,
        lambda_,
        k_i,
        e_field,
        field_es,
        field_ep,
        reflection,
        length,
        radii_curv,
        angular_opening,
    ):
        r"""
        Computes the scattered field at the intersections.

        Input
        ------
        int_point : (num_active_samples, 3), np.float_
            Positions at which the rays intersect with the scene

        objects : (num_active_samples,), np.int_
            Indices of the intersected objects

        normals : (num_active_samples, 3), np.float_
            Normals to the intersected primitives

        etas : (num_active_samples,), np.complex_
            Relative permittivities of the intersected primitives

        scattering_coefficient : (num_active_samples,), np.float_
            Scattering coefficients of the intersected primitives

        xpd_coefficient : (num_active_samples,), np.float_
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes

        alpha_r : (num_active_samples,), np.float_
            Tensor containing the alpha_r scattering parameters of all shapes

        alpha_i : (num_active_samples,), np.float_
            Tensor containing the alpha_i scattering parameters of all shapes

        lambda_ : (num_shape,), np.float_
            Tensor containing the lambda_ scattering parameters of all shapes

        k_i : (num_active_samples, 3), np.float_
            Direction of arrival of the ray

        e_field : (num_active_samples, num_tx_patterns, 2), np.complex_
            S and P components of the incident field

        field_es : (num_active_samples, 3), np.float_
            Direction of the S component of the incident field

        field_ep : (num_active_samples, 3), np.float_
            Direction of the P component of the incident field

        reflection : bool
            Set to `True` if reflection is enabled

        length : (num_active_samples,), np.float_
            Length of the last path segment

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature

        angular_opening : (num_active_samples,), np.float_
            Angular opening

        Output
        -------
        e_field : (num_active_samples, num_tx_patterns, 2), np.complex_
            S and P components of the scattered field

        field_es : (num_active_samples, 3), np.float_
            Direction of the S component of the scattered field

        field_ep : (num_active_samples, 3), np.float_
            Direction of the P component of the scattered field

        k_s : (num_active_samples, 3), np.float_
            Direction of the scattered ray

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature of the scattered ray tube

        dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal direction of curvature of the scattered ray tube

        angular_opening : (num_active_samples,), np.float_
            Angular opening of the scattered field
        """

        # Compute and apply the spreading factor to the incident field
        # (num_active_samples,)
        sf = compute_spreading_factor(radii_curv[:, 0], radii_curv[:, 1], length)
        # (num_active_samples, 1, 1)
        num_extra_dims = max(0, e_field.ndim - sf.ndim)
        sf = sf.reshape(sf.shape + (1,) * num_extra_dims) + 0.0j
        # (num_active_samples, num_tx_patterns, 2)
        e_field *= sf

        # Represent incident field in the basis for reflection
        e_i_s, e_i_p = compute_field_unit_vectors(
            k_i, None, normals, SolverBase.EPSILON, return_e_r=False
        )

        # (num_active_samples, 2, 2)
        to_incident = component_transform(field_es, field_ep, e_i_s, e_i_p)
        # (num_active_samples, 1, 2, 2)
        to_incident = np.expand_dims(to_incident, axis=1)
        to_incident = to_incident + 0.0j
        # (num_active_samples, num_tx_patterns, 2)
        e_field_ref = matvec(to_incident, e_field)

        # Compute Fresnel reflection coefficients
        # (num_active_samples,)
        cos_theta = -dot(k_i, normals, clip=True)

        # (num_active_samples,)
        r_s, r_p = reflection_coefficient(etas, cos_theta)

        # (num_active_samples, 2)
        r = np.stack([r_s, r_p], axis=-1)
        # (num_active_samples, 1, 2)
        r = np.expand_dims(r, axis=-2)

        # Compute amplitude of the reflected field
        # (num_active_samples, num_tx_patterns)
        ref_amp = np.sqrt(np.sum(np.abs(r * e_field_ref) ** 2, axis=-1))

        # Compute incoming field and polarization vectors
        # (num_active_samples, num_tx_patterns, 1)
        e_field_s, e_field_p = np.split(e_field, 2, axis=-1)

        # (num_active_samples, 1, 3)
        field_es = np.expand_dims(field_es, axis=1)
        field_es = field_es + 0.0j
        field_ep = np.expand_dims(field_ep, axis=1)
        field_ep = field_ep + 0.0j

        # Incoming field vector
        # (num_active_samples, num_tx_patterns, 3)
        e_in = e_field_s * field_es + e_field_p * field_ep

        # Polarization vectors
        # (num_active_samples, num_tx_patterns, 3)
        e_pol_hat, _ = normalize(np.real(e_in))
        e_xpol_hat = cross(e_pol_hat, np.expand_dims(k_i, 1))

        # Compute incoming spherical unit vectors in GCS
        theta_i, phi_i = theta_phi_from_unit_vec(-k_i)
        # (num_active_samples, 1, 3)
        theta_hat_i = np.expand_dims(theta_hat(theta_i, phi_i), axis=1)
        phi_hat_i = np.expand_dims(phi_hat(phi_i), axis=1)

        # Transformation from e_pol_hat, e_xpol_hat to theta_hat_i,phi_hat_i
        # (num_active_samples, num_tx_patterns, 2, 2)
        trans_mat = component_transform(e_pol_hat, e_xpol_hat, theta_hat_i, phi_hat_i)
        trans_mat = trans_mat + 0.0j

        # Generate random phases
        # All tx_patterns get the same phases
        num_active_samples = e_field.shape[0]
        phase_shape = [num_active_samples, 1, 2]
        # (num_active_samples, 1, 2)
        phases = np.random.uniform(size=phase_shape, high=2 * np.pi)

        # Compute XPD weighting
        # (num_active_samples, 2)
        xpd_weights = np.stack(
            [np.sqrt(1 - xpd_coefficient), np.sqrt(xpd_coefficient)], axis=-1
        )
        xpd_weights = xpd_weights + 0.0j
        # (num_active_samples, 1, 2)
        xpd_weights = np.expand_dims(xpd_weights, axis=1)

        # Create scattered field components from phases and xpd_weights
        # (num_active_samples, 1, 2)
        e_field = np.exp(phases * 1.0j)
        e_field *= xpd_weights

        # Apply transformation to field vector
        # (num_active_samples, num_tx_patterns, 2)
        e_field = matvec(trans_mat, e_field)

        # Draw random directions for scattered paths
        # (num_active_samples, 3)
        k_s = sample_points_on_hemisphere(normals)

        # Evaluate scattering pattern
        # Evaluate scattering pattern for all paths.
        # If a callable is defined to compute the scattering pattern,
        # it is invoked. Otherwise, the radio materials of objects are used.
        sp_callable = self._scene.scattering_pattern_callable
        if sp_callable is None:
            # (num_active_samples,)
            f_s = ScatteringPattern.pattern(
                k_i, k_s, normals, alpha_r, alpha_i, lambda_
            )
        else:
            # (num_targets, num_sources, max_num_paths)
            f_s = sp_callable(objects, int_point, k_i, k_s, normals)

        # Compute scaled scattered field
        # (num_active_samples, num_tx_patterns, 2)
        ref_amp = np.expand_dims(ref_amp, -1)
        e_field *= ref_amp + 0.0j
        f_s = np.sqrt(f_s).reshape((-1, 1, 1))
        e_field *= f_s + 0.0j
        # Weight due to angular domain
        radii_curv += np.expand_dims(length, axis=1)
        # (num_active_samples, 1, 1)
        w = angular_opening * radii_curv[:, 0] * radii_curv[:, 1]
        num_extra_dims = max(0, e_field.ndim - w.ndim)
        w = w.reshape(w.shape + (1,) * num_extra_dims)
        # (num_active_samples, num_tx_patterns, 2)
        e_field *= np.sqrt(w).asdtype(self._dtype)

        # If reflection is enabled, then the rays are randomly
        # allocated to reflection or scattering by sampling according to the
        # scattering coefficient. An oversampling factor is applied to keep
        # differentiability with respect to the scattering coefficient.
        # This oversampling factor is the ratio between the scattering factor
        # and the (non-differientiable) probability with which a
        # scattering phenomena is selected. In our case, this probability is the
        # scattering factor.
        # If reflection is disabled, all samples are allocated to scattering to
        # maximize sample-efficiency. However, this requires correcting the
        # contribution of the reflected rays by applying the scattering factor.
        # (num_active_samples,)
        scattering_factor = scattering_coefficient + 0.0j
        # (num_active_samples, 1, 1)
        scattering_factor = scattering_factor.reshape((-1, 1, 1))
        if reflection:
            # (num_active_samples,)
            ovs_factor = np.where(scattering_factor == 0.0, 0.0, 1.0)
            # (num_active_samples, num_tx_patterns, 2)
            e_field *= ovs_factor
        else:
            # (num_active_samples, num_tx_patterns, 2)
            e_field *= scattering_factor

        # Compute outgoing spherical unit vectors in GCS
        theta_s, phi_s = theta_phi_from_unit_vec(k_s)
        # (num_active_samples, 3)
        field_es = theta_hat(theta_s, phi_s)
        field_ep = phi_hat(phi_s)

        # Update principal radii of curvature
        # (num_reflected_samples, 2)
        new_radii_curv = np.zeros_like(radii_curv)

        # Update the principal direction of curvature
        # (num_reflected_samples, 3)
        new_dir_curv_1, new_dir_curv_2 = gen_basis_from_z(k_s, SolverBase.EPSILON)
        # (num_reflected_samples, 2, 3)
        new_dirs_curv = np.stack([new_dir_curv_1, new_dir_curv_2], axis=1)

        # New angular opening
        new_angular_opening = np.full(angular_opening.shape, 2.0 * np.pi)

        return (
            e_field,
            field_es,
            field_ep,
            k_s,
            new_radii_curv,
            new_dirs_curv,
            new_angular_opening,
        )

    def _compute_ris_reflected_field(
        self,
        int_point,
        ris_ind,
        k_i,
        e_field,
        field_es,
        field_ep,
        length,
        radii_curv,
        dirs_curv,
    ):
        r"""
        Computes the field reflected by the RIS at the intersections.

        Input
        ------
        int_point : (num_active_samples, 3), np.float_
            Positions at which the rays intersect with the RIS

        ris_ind : (num_active_samples,), np.int_
            Indices of the intersected RIS

        k_i : (num_active_samples, 3), np.float_
            Direction of arrival of the ray

        e_field : (num_active_samples, num_tx_patterns, 2), np.complex_
            S and P components of the incident field

        field_es : (num_active_samples, 3), np.float_
            Direction of the S component of the incident field

        field_ep : (num_active_samples, 3), np.float_
            Direction of the P component of the incident field

        length : (num_active_samples,), np.float_
            Length of the last path segment

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature of the incident ray tube

        dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal direction of curvature of the incident ray tube

        Output
        -------
        e_field : (num_active_samples, num_tx_patterns, 2), np.complex_
            S and P components of the reflected field

        field_es : (num_active_samples, 3), np.float_
            Direction of the S component of the reflected field

        field_ep : (num_active_samples, 3), np.float_
            Direction of the P component of the reflected field

        k_s : (num_active_samples, 3), np.float_
            Direction of the reflected ray

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature of the reflected ray tube

        dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal direction of curvature of the reflected ray tube
        """
        # Compute and apply the spreading factor
        # (num_active_samples)
        sf = compute_spreading_factor(radii_curv[:, 0], radii_curv[:, 1], length)
        # (num_active_samples, 1, 1)
        num_extra_dims = max(0, e_field.ndim - sf.ndim)
        sf = sf.reshape(sf.shape + (1,) * num_extra_dims)
        sf = sf + 0.0j
        # (num_active_samples, num_tx_patterns, 2)
        e_field *= sf
        # Update radii of curvature
        # (num_active_samples, 2)
        radii_curv += np.expand_dims(length, axis=1)

        all_int_point = int_point
        all_k_i = k_i
        all_e_field = e_field
        all_field_es = field_es
        all_field_ep = field_ep
        all_radii_curv = radii_curv
        all_dirs_curv = dirs_curv

        # Outputs
        output_e_field = np.zeros([0, e_field.shape[1], 2], self._dtype)
        output_field_es = np.zeros([0, 3], np.float_)
        output_field_ep = np.zeros([0, 3], np.float_)
        output_k_r = np.zeros([0, 3], np.float_)
        output_radii_curv = np.zeros([0, 2], np.float_)
        output_dirs_curv = np.zeros([0, 2, 3], np.float_)

        # Iterate over the RIS
        for ris in self._scene.ris.values():

            # Get ID of this RIS
            this_ris_id = ris.object_id

            # Get normal of this RIS
            # (3,)
            normal = ris.world_normal
            # (1, 3)
            normal = np.expand_dims(normal, axis=0)

            # Indices of rays hitting this RIS
            # (num_active_samples,)
            this_ris_sample_ind = np.argwhere(ris_ind == this_ris_id)[:, 0]
            num_active_samples = this_ris_sample_ind.shape[0]

            # Gather incident ray directions for this RIS
            # (num_active_samples, 3)
            k_i = all_k_i[this_ris_sample_ind]

            # Boolean indicating the RIS side
            # True means it's the front, False means it's the back.
            # (num_active_samples,)
            hit_front = -np.sign(dot(k_i, normal))
            hit_front = hit_front > 0.0

            # Gather indices of rays that hit this RIS from the front
            this_ris_sample_ind = this_ris_sample_ind[np.argwhere(hit_front)[:, 0]]

            # Extract data relevant to this RIS
            # (this_ris_num_samples, 3)
            int_point = all_int_point[this_ris_sample_ind]
            # (this_ris_num_samples, 3)
            k_i = all_k_i[this_ris_sample_ind]
            # (this_ris_num_samples, num_tx_patterns, 2)
            e_field = all_e_field[this_ris_sample_ind]
            # (this_ris_num_samples, 3)
            field_es = all_field_es[this_ris_sample_ind]
            # (this_ris_num_samples, 3)
            field_ep = all_field_ep[this_ris_sample_ind]
            # (this_ris_num_samples, 2)
            radii_curv = all_radii_curv[this_ris_sample_ind]
            # (this_ris_num_samples, 2, 3)
            dirs_curv = all_dirs_curv[this_ris_sample_ind]

            # Number of rays hitting the RIS from the front
            this_ris_num_samples = k_i.shape[0]

            # Incidence phase gradient - Eq.(9)
            # (this_ris_num_samples, 3)
            grad_i = k_i - normal * dot(normal, k_i)[:, np.newaxis]
            grad_i *= -self._scene.wavenumber

            # Transform interaction points to LCS of the corresponding RIS
            # Store the rotation matrix for later
            # (1, 3, 3)
            rot_mat = rotation_matrix(ris.orientation)[np.newaxis]
            # (this_ris_num_samples, 3)
            int_point_lcs = int_point - ris.position[np.newaxis]
            int_point_lcs = matvec(rot_mat, int_point_lcs, True)

            # As the LCS assumes x=0, we can remove the first dimension
            # (this_ris_num_samples, 2)
            int_point_lcs = int_point_lcs[:, 1:]

            # Compute spatial modulation coefficient for all reradiation modes
            # gamma_m: (num_modes, this_ris_num_samples)
            # grad_m: (num_modes, this_ris_num_samples, 3)
            # hessian_m: (num_modes, this_ris_num_samples, 3, 3)
            gamma_m, grad_m, hessian_m = ris(int_point_lcs, return_grads=True)
            # Sample a single mode for each ray
            # (this_ris_num_samples,)
            mode_powers = ris.amplitude_profile.mode_powers
            mode = np.random.choice(
                mode_powers.size,
                size=this_ris_num_samples,
                p=mode_powers / np.sum(mode_powers),
            )[0]
            # gamma_m: (this_ris_num_samples,)
            # grad_m: (this_ris_num_samples, 3)
            # hessian_m: (this_ris_num_samples, 3, 3)
            gamma_m = gamma_m.T[mode]
            grad_m = np.transpose(grad_m, [1, 0, 2])[mode]
            hessian_m = np.transpose(hessian_m, [1, 0, 2, 3])[mode]
            # Bring RIS phase gradient to GCS
            # (this_ris_num_samples, 3)
            grad_m = matvec(rot_mat, grad_m)

            # Bring RIS phase Hessian to GCS
            # (this_ris_num_samples, 3, 3)
            hessian_m = rot_mat @ (hessian_m @ np.moveaxis(rot_mat, -1, -2))

            # Compute total phase gradient - Eq.(11)
            # (this_ris_num_samples, 3)
            grad = grad_i + grad_m

            # Compute direction of reflected ray - Eq.(13)
            # (this_ris_num_samples, 3)
            k_r = -grad / self._scene.wavenumber
            k_r += np.sqrt(1 - np.sum(k_r**2, axis=-1, keepdims=True)) * normal
            # Compute linear transformation operator - Eq.(22)
            # (this_ris_num_samples, 3, 3)
            l = (
                -outer(k_r, normal)
                / np.sum(k_r * normal, axis=-1, keepdims=True)[..., np.newaxis]
            )
            l += np.tile(np.eye(3, dtype=l.dtype), [l.shape[0], 1, 1])

            # Compute incident curvature matrix - Eq.(4)
            # (this_ris_num_samples, 3, 3)
            q_i = (
                1.0
                / radii_curv[:, 0][:, np.newaxis]
                * outer(dirs_curv[:, 0], dirs_curv[:, 0])
            )
            q_i += (
                1.0
                / radii_curv[:, 1][:, np.newaxis]
                * outer(dirs_curv[:, 1], dirs_curv[:, 1])
            )

            # Compute reflected curvature matrix - Eq.(21)
            # (this_ris_num_samples, 3, 3)
            q_r = q_i - 1 / self._scene.wavenumber * hessian_m @ l
            q_r = np.moveaxis(l, -1, -2) @ q_r

            # Extract principal axes of curvature and associated radii - Eq.(4)
            e, v, _ = np.linalg.svd(q_r)
            # (this_ris_num_samples, 2)
            radii_curv = 1 / e[:, :2]
            # (this_ris_num_samples, 2, 3)
            dirs_curv = np.transpose(v[..., :2], [0, 2, 1])

            # Basis vectors for incoming field
            # (this_ris_num_samples, 3)
            theta_i, phi_i = theta_phi_from_unit_vec(k_i)
            e_i_s = theta_hat(theta_i, phi_i)
            e_i_p = phi_hat(phi_i)

            # Component transform
            # (this_ris_num_samples, 1, 2, 2)
            mat_comp = component_transform(field_es, field_ep, e_i_s, e_i_p)
            mat_comp = mat_comp + 0.0j
            mat_comp = mat_comp[:, np.newaxis]

            # Outgoing field - Eq.(14)
            # (this_ris_num_samples, num_tx_patterns, 2)
            e_field = matvec(mat_comp, e_field)
            e_field *= gamma_m[:, np.newaxis, np.newaxis]

            # Basis vectors for reflected field
            # (this_ris_num_samples, 3)
            theta_r, phi_r = theta_phi_from_unit_vec(k_r)
            field_es = theta_hat(theta_r, phi_r)
            field_ep = phi_hat(phi_r)

            # Concatenate rays from reflection by all RIS
            # and create all-zeros samples for the inactive rays
            # which will be dropped in a later stage.
            n_p = num_active_samples - this_ris_num_samples

            def pad(x, n_p):
                """Pad input tensor with n-p zero samples"""
                paddings = np.concatenate(
                    [[[0, n_p]], np.zeros([x.ndim - 1, 2], np.int_)], axis=0
                )
                return np.pad(x, paddings)

            output_e_field = np.concatenate([output_e_field, pad(e_field, n_p)], axis=0)
            output_field_es = np.concatenate(
                [output_field_es, pad(field_es, n_p)], axis=0
            )
            output_field_ep = np.concatenate(
                [output_field_ep, pad(field_ep, n_p)], axis=0
            )
            output_k_r = np.concatenate([output_k_r, pad(k_r, n_p)], axis=0)
            output_radii_curv = np.concatenate(
                [output_radii_curv, pad(radii_curv, n_p)], axis=0
            )
            output_dirs_curv = np.concatenate(
                [output_dirs_curv, pad(dirs_curv, n_p)], axis=0
            )

        output = (
            output_e_field,
            output_field_es,
            output_field_ep,
            output_k_r,
            output_radii_curv,
            output_dirs_curv,
        )
        return output

    def _init_e_field(self, valid_ray, samples_tx_indices, k_tx, tx_rot_mat):
        r"""
        Initialize the electric field for the rays flagged as valid.

        Input
        -----
        valid_ray : (num_samples,), np.bool_
            Flag set to `True` if the ray is valid

        samples_tx_indices : (num_samples,), np.int_
            Index of the transmitter from which the ray originated

        k_tx : (num_samples, 3). np.float_
            Direction of departure

        tx_rot_mat : (num_tx, 3, 3), np.float_
            Matrix to go transmitter LCS to the GCS

        Output
        -------
        e_field : (num_valid_samples, num_tx_patterns, 2), np.complex_
            Emitted electric field S and P components

        field_es : (num_valid_samples, 3), np.float_
            Direction of the S component of the electric field

        field_ep : (num_valid_samples, 3), np.float_
            Direction of the P component of the electric field
        """

        num_samples = valid_ray.shape[0]
        # (num_valid_samples,)
        valid_ind = np.argwhere(valid_ray)[:, 0]
        # (num_valid_samples,)
        valid_tx_ind = samples_tx_indices[valid_ind]
        # (num_valid_samples, 3)
        k_tx = k_tx[valid_ind]
        # (num_valid_samples, 3, 3)
        tx_rot_mat = tx_rot_mat[valid_tx_ind]

        # val_e_field : (num_valid_samples, num_tx_patterns, 2)
        # val_field_es, val_field_ep : (num_valid_samples, 3)
        val_e_field, val_field_es, val_field_ep = self._compute_antenna_patterns(
            tx_rot_mat, self._scene.tx_array.antenna.patterns, k_tx
        )
        valid_ind = np.expand_dims(valid_ind, axis=-1)
        # (num_samples, num_tx_patterns, 2)
        e_field = np.zeros([num_samples, val_e_field.shape[1], 2], self._dtype)
        e_field[valid_ind] = val_e_field
        # (num_samples, 3)
        field_es = np.zeros([num_samples, 3], np.float_)
        field_es[valid_ind] = val_field_es
        field_ep = np.zeros([num_samples, 3], np.float_)
        field_ep[valid_ind] = val_field_ep

        return e_field, field_es, field_ep

    def _extract_active_ris_rays(
        self,
        active_ind,
        int_point,
        previous_int_point,
        primitives,
        e_field,
        field_es,
        field_ep,
        radii_curv,
        dirs_curv,
        angular_opening,
    ):
        r"""
        Extracts the active rays hitting a RIS.

        Input
        ------
        active_ind : (num_active_samples,), np.int_
            Indices of the active rays

        int_point : (num_samples, 3), np.float_
            Positions at which the rays intersect with the scene. For the rays
            that did not intersect the scene, the corresponding position should
            be ignored.

        previous_int_point : (num_samples, 3), np.float_
            Positions of the previous intersection points of the rays with
            the scene

        primitives : (num_samples,), np.int_
            Indices of the intersected primitives

        e_field : (num_samples, num_tx_patterns, 2), np.complex_
            S and P components of the electric field

        field_es : (num_samples, 3), np.float_
            Direction of the S component of the field

        field_ep : (num_samples, 3), np.float_
            Direction of the P component of the field

        radii_curv : (num_samples, 2), np.float_
            Principal radii of curvature of the ray tubes

        dirs_curv : (num_samples, 2, 3), np.float_
            Principal directions of curvature of the ray tubes

        angular_opening : (num_active_samples,), np.float_
            Angular opening

        Output
        -------
        act_e_field : (num_active_samples, num_tx_patterns, 2), np.complex_
            S and P components of the electric field of the active rays

        act_field_es : (num_active_samples, 3), np.float_
            Direction of the S component of the field of the active rays

        act_field_ep : (num_active_samples, 3), np.float_
            Direction of the P component of the field of the active rays

        act_point : (num_active_samples, 3), np.float_
            Positions at which the rays intersect with the scene

        act_k_i : (num_active_samples, 3), np.float_
            Direction of the active incident ray

        act_dist : (num_active_samples,), np.float_
            Length of the last path segment, i.e., distance between `int_point`
            and `previous_int_point`

        act_radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature of the ray tubes

        act_dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal directions of curvature of the ray tubes

        act_angular_opening : (num_active_samples,), np.float_
            Angular opening
        """

        # Extract the rays that interact the scene
        # (num_active_samples, num_tx_patterns, 2)
        act_e_field = e_field[active_ind]
        # (num_active_samples, 3)
        act_field_es = field_es[active_ind]
        # (num_active_samples, 3)
        act_field_ep = field_ep[active_ind]
        # (num_active_samples, 2)
        act_radii_curv = radii_curv[active_ind]
        # (num_active_samples, 2, 3)
        act_dirs_curv = dirs_curv[active_ind]
        # (num_active_samples, 3)
        act_previous_int_point = previous_int_point[active_ind]
        # Current intersection point
        # (num_active_samples, 3)
        int_point = int_point[active_ind]
        # (num_active_samples,)
        act_primitives = primitives[active_ind]

        # Direction of arrival
        # (num_active_samples, 3)
        act_k_i, act_dist = normalize(int_point - act_previous_int_point)

        # Extract angular openings
        act_angular_opening = angular_opening[active_ind]

        output = (
            act_e_field,
            act_field_es,
            act_field_ep,
            int_point,
            act_k_i,
            act_dist,
            act_radii_curv,
            act_dirs_curv,
            act_primitives,
            act_angular_opening,
        )

        return output

    def _extract_active_rays(
        self,
        active_ind,
        int_point,
        previous_int_point,
        primitives,
        e_field,
        field_es,
        field_ep,
        etas,
        scattering_coefficient,
        xpd_coefficient,
        alpha_r,
        alpha_i,
        lambda_,
        radii_curv,
        dirs_curv,
        angular_opening,
    ):
        r"""
        Extracts the active rays.

        Input
        ------
        active_ind : (num_active_samples,), np.int_
            Indices of the active rays

        int_point : (num_samples, 3), np.float_
            Positions at which the rays intersect with the scene. For the rays
            that did not intersect the scene, the corresponding position should
            be ignored.

        previous_int_point : (num_samples, 3), np.float_
            Positions of the previous intersection points of the rays with
            the scene

        primitives : (num_samples), np.int_
            Indices of the intersected primitives

        e_field : (num_samples, num_tx_patterns, 2), np.complex_
            S and P components of the electric field

        field_es : (num_samples, 3), np.float_
            Direction of the S component of the field

        field_ep : (num_samples, 3), np.float_
            Direction of the P component of the field

        etas : (num_shape), np.complex_ | `None`
            Tensor containing the complex relative permittivities of all shapes

        scattering_coefficient : (num_shape), np.float_ | `None`
            Tensor containing the scattering coefficients of all shapes

        xpd_coefficient : (num_shape), np.float_ | `None`
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes

        alpha_r : (num_shape), np.float_ | `None`
            Tensor containing the alpha_r scattering parameters of all shapes

        alpha_i : (num_shape), np.float_ | `None`
            Tensor containing the alpha_i scattering parameters of all shapes

        lambda_ : (num_shape), np.float_ | `None`
            Tensor containing the lambda_ scattering parameters of all shapes

        radii_curv : (num_samples, 2), np.float_
            Principal radii of curvature of the ray tubes

        dirs_curv : (num_samples, 2, 3), np.float_
            Principal directions of curvature of the ray tubes

        angular_opening : (num_active_samples), np.float_
            Angular opening

        Output
        -------
        act_e_field : (num_active_samples, num_tx_patterns, 2), np.complex_
            S and P components of the electric field of the active rays

        act_field_es : (num_active_samples, 3), np.float_
            Direction of the S component of the field of the active rays

        act_field_ep : (num_active_samples, 3), np.float_
            Direction of the P component of the field of the active rays

        act_point : (num_active_samples, 3), np.float_
            Positions at which the rays intersect with the scene

        act_normals : (num_active_samples, 3), np.float_
            Normals at the intersection point. The normals are oriented to match
            the direction opposite to the incident ray

        act_etas : (num_active_samples,), np.complex_
            Relative permittivity of the intersected primitives

        act_scat_coeff : (num_active_samples,), np.float_
            Scattering coefficient of the intersected primitives

        act_k_i : (num_active_samples, 3), np.float_
            Direction of the active incident ray

        act_xpd_coefficient : (num_active_samples,), np.float_ | `None`
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes.
            Only returned if ``xpd_coefficient`` is not `None`.

        act_alpha_r : (num_active_samples,), np.float_
            Tensor containing the alpha_r scattering parameters of all shapes.
            Only returned if ``alpha_r`` is not `None`.

        act_alpha_i : (num_active_samples,), np.float_
            Tensor containing the alpha_i scattering parameters of all shapes
            Only returned if ``alpha_i`` is not `None`.

        act_lambda_ : (num_active_samples,), np.float_
            Tensor containing the lambda_ scattering parameters of all shapes
            Only returned if ``lambda_`` is not `None`.

        act_objects : (num_active_samples,), np.int_
            Indices of the intersected objects

        act_dist : (num_active_samples,), np.float_
            Length of the last path segment, i.e., distance between `int_point`
            and `previous_int_point`

        act_radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature of the ray tubes

        act_dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal directions of curvature of the ray tubes

        act_primitives : (num_active_samples,), np.int_
            Indices of the intersected primitives

        act_angular_opening : (num_active_samples,), np.float_
            Angular opening
        """

        # Extract the rays that interact the scene
        # (num_active_samples, num_tx_patterns, 2)
        act_e_field = e_field[active_ind]
        # (num_active_samples, 3)
        act_field_es = field_es[active_ind]
        # (num_active_samples, 3)
        act_field_ep = field_ep[active_ind]
        # (num_active_samples, 2)
        act_radii_curv = radii_curv[active_ind]
        # (num_active_samples, 2, 3)
        act_dirs_curv = dirs_curv[active_ind]
        # (num_active_samples, 3)
        act_previous_int_point = previous_int_point[active_ind]
        # Current intersection point
        # (num_active_samples, 3)
        int_point = int_point[active_ind]
        # (num_active_samples,)
        act_primitives = primitives[active_ind]
        # (num_active_samples,)
        act_objects = self._primitives_2_objects[act_primitives]

        # Extract the normals to the intersected primitives
        # (num_active_samples, 3)
        if self._normals.shape[0] > 0:
            act_normals = self._normals[act_primitives]
        else:
            act_normals = None

        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self._scene.radio_material_callable
        if rm_callable is None:
            # Extract the material properties of the intersected objects
            if etas is not None:
                # (num_active_samples,)
                act_etas = etas[act_objects]
            else:
                act_etas = None
            if scattering_coefficient is not None:
                # (num_active_samples,)
                act_scat_coeff = scattering_coefficient[act_objects]
            else:
                act_scat_coeff = None
            if xpd_coefficient is not None:
                # (num_active_samples)
                act_xpd_coefficient = xpd_coefficient[act_objects]
            else:
                act_xpd_coefficient = None
        else:
            # (num_active_samples,)
            act_etas, act_scat_coeff, act_xpd_coefficient = rm_callable(
                act_objects, int_point
            )

        # If no callable is defined for the scattering pattern, we need to
        # extract the properties of the scattering patterns built-in Sionna
        if (self._scene.scattering_pattern_callable is None) and (alpha_r is not None):
            # (num_active_samples,)
            act_alpha_r = alpha_r[act_objects]
            act_alpha_i = alpha_i[act_objects]
            act_lambda_ = lambda_[act_objects]
        else:
            act_alpha_r = act_alpha_i = act_lambda_ = None

        # Direction of arrival
        # (num_active_samples, 3)
        act_k_i, act_dist = normalize(int_point - act_previous_int_point)

        # Ensure the normal points in the direction -k_i
        if act_normals is not None:
            # (num_active_samples, 1)
            flip_normal = -np.sign(dot(act_k_i, act_normals, keepdim=True))
            # (num_active_samples, 3)
            act_normals = flip_normal * act_normals

        # Extract angular openings
        act_angular_opening = angular_opening[active_ind]

        output = (
            act_e_field,
            act_field_es,
            act_field_ep,
            int_point,
            act_normals,
            act_etas,
            act_scat_coeff,
            act_k_i,
            act_xpd_coefficient,
            act_alpha_r,
            act_alpha_i,
            act_lambda_,
            act_objects,
            act_dist,
            act_radii_curv,
            act_dirs_curv,
            act_primitives,
            act_angular_opening,
        )

        return output

    def _sample_interaction_phenomena(
        self,
        active,
        int_point,
        primitives,
        scattering_coefficient,
        reflection,
        scattering,
    ):
        r"""
        Samples the interaction phenomena to apply to each active ray, among
        scattering or reflection.

        This is done by sampling a Bernouilli distribution with probability p
        equal to the square of the scattering coefficient amplitude, as it
        corresponds to the ratio of the reflected energy that goes to
        scattering. With probability p, the ray is scattered. Otherwise, it is
        reflected.

        Input
        ------
        active : (num_samples,), np.bool_
            Flag indicating if a ray is active

        int_point : (num_samples, 3), np.float_
            Positions at which the rays intersect with the scene. For the rays
            that did not intersect the scene, the corresponding position should
            be ignored.

        scattering_coefficient : (num_shape,), np.complex_
            Scattering coefficients of all shapes

        reflection : bool
            Set to `True` if reflection is enabled

        scattering : bool
            Set to `True` if scattering is enabled

        Output
        -------
        reflect_ind : (num_reflected_samples,), np.int_
            Indices of the rays that are reflected

        scatter_ind : (num_scattered_samples,), np.int_
            Indices of the rays that are scattered
        """

        # Indices of the active samples
        # (num_active_samples,)
        active_ind = np.argwhere(active)[:, 0]

        # If only one of reflection or scattering is enabled, then all the
        # samples are used for the enabled phenomena to avoid wasting samples
        # by allocating them to a phenomena that is not requested by the users.
        # This approach, however, requires to correct later the contribution
        # of the rays by weighting them by the square of the scattering or
        # reduction factor, depending on the selected phenomena.
        # This is done in the functions that compute the reflected and scattered
        # field.
        if not (reflection or scattering):
            reflect_ind = np.zeros([0], np.int_)
            scatter_ind = np.zeros([0], np.int_)
        elif not reflection:
            reflect_ind = np.zeros([0], np.int_)
            scatter_ind = active_ind
        elif not scattering:
            reflect_ind = active_ind
            scatter_ind = np.zeros([0], np.int_)
        else:
            # Scattering coefficients of the intersected objects
            # (num_active_samples,)
            act_primitives = primitives[active_ind]
            act_objects = self._primitives_2_objects[act_primitives]
            # Current intersection point
            # (num_active_samples, 3)
            int_point = int_point[active_ind]

            # If a callable is defined to compute the radio material properties,
            # it is invoked. Otherwise, the radio materials of objects are used.
            rm_callable = self._scene.radio_material_callable
            if rm_callable is None:
                # (num_active_samples,)
                act_scat_coeff = scattering_coefficient[act_objects]
            else:
                # (num_active_samples,)
                _, act_scat_coeff, _ = rm_callable(act_objects, int_point)

            # Probability of scattering
            # (num_active_samples)
            prob_scatter = np.square(np.abs(act_scat_coeff))

            # Sampling a Bernoulli distribution
            # (num_active_samples)
            scatter = np.random.uniform(
                prob_scatter.shape,
                np.zeros((), np.float_),
                np.ones((), np.float_),
                dtype=np.float_,
            )
            scatter = scatter < prob_scatter

            # Extract indices of the reflected and scattered rays
            # (num_reflected_samples,)
            reflect_ind = active_ind[np.argwhere(~scatter)[:, 0]]
            # (num_scattered_samples,)
            scatter_ind = active_ind[np.argwhere(scatter)[:, 0]]

        return reflect_ind, scatter_ind

    def _apply_reflection(
        self,
        active_ind,
        int_point,
        previous_int_point,
        primitives,
        e_field,
        field_es,
        field_ep,
        etas,
        scattering_coefficient,
        scattering,
        radii_curv,
        dirs_curv,
        angular_opening,
    ):
        r"""
        Apply reflection.

        Input
        ------
        active_ind : (num_reflected_samples,), np.int_
            Indices of the *active* rays to which reflection must be applied.

        int_point : (num_samples, 3), np.float_
            Locations of the intersection point

        previous_int_point : (num_samples, 3), np.float_
            Locations of the intersection points of the previous interaction.

        primitives : (num_samples,), np.int_
            Indices of the intersected primitives

        e_field : (num_samples, num_tx_patterns, 2), np.complex_
            S and P components of the electric field

        field_es : (num_samples, 3), np.float_
            Direction of the S component of the field

        field_ep : (num_samples, 3), np.float_
            Direction of the P component of the field

        etas : (num_shape,), np.complex_
            Complex relative permittivities of all shapes

        scattering_coefficient : (num_shape,), np.float_
            Scattering coefficients of all shapes

        scattering : bool
            Set to `True` if scattering is enabled

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature

        dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal direction of curvature

        angular_opening : (num_active_samples,), np.float_
            Angular opening

        Output
        -------
        e_field : (num_reflected_samples, num_tx_patterns, 2), np.complex_
            S and P components of the reflected electric field

        field_es : (num_reflected_samples, 3), np.float_
            Direction of the S component of the reflected field

        field_ep : (num_reflected_samples, 3), np.float_
            Direction of the P component of the reflected field

        int_point : (num_reflected_samples, 3), np.float_
            Locations of the intersection point

        k_r : (num_reflected_samples, 3), np.float_
            Direction of the reflected ray

        radii_curv : (num_reflected_samples, 2), np.float_
            Principal radii of curvature of the reflected field

        dirs_curv : (num_reflected_samples, 2, 3), np.float_
            Principal direction of curvature of the reflected field

        angular_opening : (num_reflected_samples,), np.float_
            Angular opening of the reflected ray
        """

        # Prepare field computation
        # This function extract the data for the rays to which reflection
        # must be applied, and ensures that the normals are correctly oriented.
        act_data = self._extract_active_rays(
            active_ind,
            int_point,
            previous_int_point,
            primitives,
            e_field,
            field_es,
            field_ep,
            etas,
            scattering_coefficient,
            None,
            None,
            None,
            None,
            radii_curv,
            dirs_curv,
            angular_opening,
        )
        # (num_reflected_samples, num_tx_patterns, 2)
        e_field = act_data[0]
        # (num_reflected_samples, 3)
        field_es = act_data[1]
        field_ep = act_data[2]
        int_point = act_data[3]
        # (num_reflected_samples, 3)
        act_normals = act_data[4]
        # (num_reflected_samples,)
        act_etas = act_data[5]
        act_scat_coeff = act_data[6]
        # (num_reflected_samples, 3)
        k_i = act_data[7]
        # Length of the last path segment
        # (num_reflected_samples,)
        length = act_data[13]
        # Principal radii and directions of curvatures
        # (num_reflected_samples, 2)
        radii_curv = act_data[14]
        # (num_reflected_samples, 2, 3)
        dirs_curv = act_data[15]
        # (num_reflected_samples)
        angular_opening = act_data[17]

        # Compute the reflected field
        e_field, field_es, field_ep, k_r, radii_curv, dirs_curv = (
            self._compute_reflected_field(
                act_normals,
                act_etas,
                act_scat_coeff,
                k_i,
                e_field,
                field_es,
                field_ep,
                scattering,
                length,
                radii_curv,
                dirs_curv,
            )
        )

        output = (
            e_field,
            field_es,
            field_ep,
            int_point,
            k_r,
            radii_curv,
            dirs_curv,
            angular_opening,
        )
        return output

    def _apply_scattering(
        self,
        active_ind,
        int_point,
        previous_int_point,
        primitives,
        e_field,
        field_es,
        field_ep,
        etas,
        scattering_coefficient,
        xpd_coefficient,
        alpha_r,
        alpha_i,
        lambda_,
        reflection,
        radii_curv,
        dirs_curv,
        angular_opening,
    ):
        r"""
        Apply scattering.

        Input
        ------
        active_ind : (num_scattered_samples,), np.int_
            Indices of the *active* rays to which scattering must be applied.

        int_point : (num_samples, 3), np.float_
            Locations of the intersection point

        previous_int_point : (num_samples, 3), np.float_
            Locations of the intersection points of the previous interaction.

        primitives : (num_samples), np.int_
            Indices of the intersected primitives

        e_field : (num_samples, num_tx_patterns, 2), np.complex_
            S and P components of the electric field

        field_es : (num_samples, 3), np.float_
            Direction of the S component of the field

        field_ep : (num_samples, 3), np.float_
            Direction of the P component of the field

        etas : (num_shape,), np.complex
            Complex relative permittivities of all shapes

        scattering_coefficient : (num_shape,), np.float_
            Scattering coefficients of all shapes

        xpd_coefficient : (num_shape,), np.float_ | `None`
            Cross-polarization discrimination coefficients of all shapes

        alpha_r : (num_shape,), np.float_ | `None`
            alpha_r scattering parameters of all shapes

        alpha_i : (num_shape,), np.float_ | `None`
            alpha_i scattering parameters of all shapes

        lambda_ : (num_shape,), np.float_ | `None`
            lambda_ scattering parameters of all shapes

        reflection : bool
            Set to `True` if reflection is enabled

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature

        dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal direction of curvature

        angular_opening : (num_active_samples,), np.float_
            Angular opening

        Output
        -------
        e_field : (num_scattered_samples, num_tx_patterns, 2), np.complex_
            S and P components of the scattered electric field

        field_es : (num_scattered_samples, 3), np.float_
            Direction of the S component of the scattered field

        field_ep : (num_scattered_samples, 3), np.float_
            Direction of the P component of the scattered field

        int_point : (num_scattered_samples, 3), np.float_
            Locations of the intersection point

        k_r : (num_scattered_samples, 3), np.float_
            Direction of the scattered ray

        radii_curv : (num_scattered_samples, 2), np.float_
            Principal radii of curvature of the scattered field

        dirs_curv : (num_scattered_samples, 2, 3), np.float_
            Principal direction of curvature of the scattered field

        angular_opening : (num_scattered_samples,), np.float_
            Angular opening of the scattered field
        """

        # Prepare field computation
        # This function extract the data for the rays to which scattering
        # must be applied, and ensures that the normals are correcly oriented.
        act_data = self._extract_active_rays(
            active_ind,
            int_point,
            previous_int_point,
            primitives,
            e_field,
            field_es,
            field_ep,
            etas,
            scattering_coefficient,
            xpd_coefficient,
            alpha_r,
            alpha_i,
            lambda_,
            radii_curv,
            dirs_curv,
            angular_opening,
        )
        # (num_scattered_samples, num_tx_patterns, 2)
        e_field = act_data[0]
        # (num_scattered_samples, 3)
        field_es = act_data[1]
        field_ep = act_data[2]
        int_point = act_data[3]
        # (num_scattered_samples, 3)
        act_normals = act_data[4]
        # (num_scattered_samples,)
        act_etas = act_data[5]
        act_scat_coeff = act_data[6]
        # (num_scattered_samples, 3)
        k_i = act_data[7]
        # (num_scattered_samples,)
        act_xpd_coefficient = act_data[8]
        act_alpha_r = act_data[9]
        act_alpha_i = act_data[10]
        act_lambda_ = act_data[11]
        act_objects = act_data[12]
        # Length of the last path segment
        # (num_scattered_samples,)
        length = act_data[13]
        # Principal radii and directions of curvatures
        # (num_scattered_samples, 2)
        radii_curv = act_data[14]
        # (num_scattered_samples, 2, 3)
        dirs_curv = act_data[15]
        # (num_scattered_samples)
        angular_opening = act_data[17]

        # Compute the scattered field
        e_field, field_es, field_ep, k_r, radii_curv, dirs_curv, angular_opening = (
            self._compute_scattered_field(
                int_point,
                act_objects,
                act_normals,
                act_etas,
                act_scat_coeff,
                act_xpd_coefficient,
                act_alpha_r,
                act_alpha_i,
                act_lambda_,
                k_i,
                e_field,
                field_es,
                field_ep,
                reflection,
                length,
                radii_curv,
                angular_opening,
            )
        )

        output = (
            e_field,
            field_es,
            field_ep,
            int_point,
            k_r,
            radii_curv,
            dirs_curv,
            angular_opening,
        )
        return output

    def _apply_ris_reflection(
        self,
        active_ind,
        int_point,
        previous_int_point,
        primitives,
        e_field,
        field_es,
        field_ep,
        radii_curv,
        dirs_curv,
        angular_opening,
    ):
        r"""
        Apply scattering.

        Input
        ------
        active_ind : (num_ris_reflected_samples,), np.int_
            Indices of the *active* rays to which scattering must be applied.

        int_point : (num_samples, 3), np.float_
            Locations of the intersection point

        previous_int_point : (num_samples, 3), np.float_
            Locations of the intersection points of the previous interaction.

        primitives : (num_samples,), np.int_
            Indices of the intersected primitives

        e_field : (num_samples, num_tx_patterns, 2), np.complex_
            S and P components of the electric field

        field_es : (num_samples, 3), np.float_
            Direction of the S component of the field

        field_ep : (num_samples, 3), np.float_
            Direction of the P component of the field

        radii_curv : (num_active_samples, 2), np.float_
            Principal radii of curvature

        dirs_curv : (num_active_samples, 2, 3), np.float_
            Principal direction of curvature

        angular_opening : (num_active_samples,), np.float_
            Angular opening

        Output
        -------
        e_field : (num_ris_reflected_samples, num_tx_patterns, 2), np.complex_
            S and P components of the reflected electric field

        field_es : (num_ris_reflected_samples, 3), np.float_
            Direction of the S component of the reflected field

        field_ep : (num_ris_reflected_samples, 3), np.float_
            Direction of the P component of the reflected field

        int_point : (num_ris_reflected_samples, 3), np.float_
            Locations of the intersection point

        k_r : (num_ris_reflected_samples, 3), np.float_
            Direction of the reflected ray

        radii_curv : (num_ris_reflected_samples, 2), np.float_
            Principal radii of curvature of the reflected field

        dirs_curv : (num_ris_reflected_samples, 2, 3), np.float_
            Principal direction of curvature of the reflected field

        angular_opening : (num_ris_reflected_samples,), np.float_
            Angular opening of the reflected field
        """
        # Prepare field computation
        # This function extract the data for the rays to which scattering
        # must be applied, and ensures that the normals are correctly oriented.
        act_data = self._extract_active_ris_rays(
            active_ind,
            int_point,
            previous_int_point,
            primitives,
            e_field,
            field_es,
            field_ep,
            radii_curv,
            dirs_curv,
            angular_opening,
        )
        # (num_ris_reflected_samples, num_tx_patterns, 2)
        e_field = act_data[0]
        # (num_ris_reflected_samples, 3)
        field_es = act_data[1]
        field_ep = act_data[2]
        int_point = act_data[3]
        # (num_ris_reflected_samples, 3)
        k_i = act_data[4]
        # Length of the last path segment
        # (num_ris_reflected_samples)
        length = act_data[5]
        # Principal radii and directions of curvatures
        # (num_ris_reflected_samples, 2)
        radii_curv = act_data[6]
        # (num_ris_reflected_samples, 2, 3)
        dirs_curv = act_data[7]
        # (num_ris_reflected_samples,)
        ris_ind = act_data[8]
        # (num_ris_reflected_samples,)
        angular_opening = act_data[9]

        # Compute the reflected field
        e_field, field_es, field_ep, k_r, radii_curv, dirs_curv = (
            self._compute_ris_reflected_field(
                int_point,
                ris_ind,
                k_i,
                e_field,
                field_es,
                field_ep,
                length,
                radii_curv,
                dirs_curv,
            )
        )

        output = (
            e_field,
            field_es,
            field_ep,
            int_point,
            k_r,
            radii_curv,
            dirs_curv,
            angular_opening,
        )
        return output

    def _shoot_and_bounce(
        self,
        meas_plane,
        ris_scene,
        rx_orientation,
        sources_positions,
        sources_orientations,
        max_depth,
        num_samples,
        combining_vec,
        precoding_vec,
        cm_center,
        cm_orientation,
        cm_size,
        cm_cell_size,
        los,
        reflection,
        diffraction,
        scattering,
        ris,
        etas,
        scattering_coefficient,
        xpd_coefficient,
        alpha_r,
        alpha_i,
        lambda_,
    ):
        r"""
        Runs shoot-and-bounce to build the coverage map for LoS, reflection,
        and scattering.

        If ``diffraction`` is set to `True`, this function also returns the
        primitives in LoS with at least one transmitter.

        Input
        ------
        meas_plane : mi.Shape
            Mitsuba rectangle defining the measurement plane

        ris_scene : mi.Scene
            Mistuba scene containing the RIS

        rx_orientation : (3,), np.float_
            Orientation of the receiver.

        sources_positions : (num_tx, 3], np.float_
            Coordinates of the sources.

        max_depth : int
            Maximum number of reflections

        num_samples : int
            Number of rays initially shooted from the transmitters.
            This number is shared by all transmitters, i.e.,
            ``num_samples/num_tx`` are shooted for each transmitter.

        combining_vec : (num_rx_ant,), np.complex_
            Combining vector.
            If set to `None`, then no combining is applied, and
            the energy received by all antennas is summed.

        precoding_vec : (num_tx or 1, num_tx_ant], np.complex_
            Precoding vectors of the transmitters

        cm_center : (3,), np.float_
            Center of the coverage map

        cm_orientation : (3,), np.float_
            Orientation of the coverage map

        cm_size : (2,), np.float_
            Scale of the coverage map.
            The width of the map (in the local X direction) is scale[0]
            and its map (in the local Y direction) scale[1].

        cm_cell_size : (2,), np.float_
            Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            meters of a cell of the coverage map

        los : bool
            If set to `True`, then the LoS paths are computed.

        reflection : bool
            If set to `True`, then the reflected paths are computed.

        diffraction : bool
            If set to `True`, then the diffracted paths are computed.

        scattering : bool
            If set to `True`, then the scattered paths are computed.

        ris : bool
            If set to `True`, then paths involving RIS are computed.

        etas : (num_shape,), np.complex_
            Tensor containing the complex relative permittivities of all shapes

        scattering_coefficient : (num_shape,), np.float_
            Tensor containing the scattering coefficients of all shapes

        xpd_coefficient : (num_shape,), np.float_
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes

        alpha_r : (num_shape,), np.float_
            Tensor containing the alpha_r scattering parameters of all shapes

        alpha_i : (num_shape,), np.float_
            Tensor containing the alpha_i scattering parameters of all shapes

        lambda_ : (num_shape,), np.float_
            Tensor containing the lambda_ scattering parameters of all shapes

        Output
        ------
        cm : (num_tx, num_cells_y, num_cells_x), np.float_
            Coverage map for every transmitter.
            Includes LoS, reflection, and scattering.

        los_primitives: [num_los_primitives], int | `None`
            Primitives in LoS.
            `None` is returned if ``diffraction`` is set to `False`.
        """

        # Ensure that sample count can be distributed over the emitters
        num_tx = sources_positions.shape[0]
        samples_per_tx_float = np.ceil(num_samples / num_tx)
        samples_per_tx = int(samples_per_tx_float)
        num_samples = num_tx * samples_per_tx

        # Transmitters and receivers rotation matrices
        # (3, 3)
        rx_rot_mat = rotation_matrix(rx_orientation)
        # (num_tx, 3, 3)
        tx_rot_mat = rotation_matrix(sources_orientations)

        # Rotation matrix to go from the measurement plane LCS to the GCS, and
        # the othwer way around
        # (3,3)
        rot_mp_2_gcs = rotation_matrix(cm_orientation)
        rot_gcs_2_mp = rot_mp_2_gcs.T
        # Normal to the CM
        # Add a dimension for broadcasting
        # (1, 3)
        cm_normal = np.expand_dims(rot_mp_2_gcs[:, 2], axis=0)

        # Number of cells in the coverage map
        # (2,2)
        num_cells_x = np.ceil(cm_size[0] / cm_cell_size[0]).astype(np.int_)
        num_cells_y = np.ceil(cm_size[1] / cm_cell_size[1]).astype(np.int_)
        num_cells = np.stack([num_cells_x, num_cells_y], axis=-1)

        # Primitives in LoS are required for diffraction
        los_primitives = None

        # Initialize rays.
        # Direction arranged in a Fibonacci lattice on the unit
        # sphere.
        # (num_samples, 3)
        ps = fibonacci_lattice(samples_per_tx, np.float_)
        ps = np.tile(ps, [num_tx, 1])
        ps_dr = self._mi_point2_t(ps)
        k_tx_dr = mi.warp.square_to_uniform_sphere(ps_dr)
        k_tx = mi_to_np_ndarray(k_tx_dr, np.float_)
        # Origin placed on the given transmitters
        # (num_samples,)
        samples_tx_indices_dr = dr.linspace(
            self._mi_scalar_t, 0, num_tx - 1e-7, num=num_samples, endpoint=False
        )
        samples_tx_indices_dr = mi.Int32(samples_tx_indices_dr)
        samples_tx_indices = mi_to_np_ndarray(samples_tx_indices_dr, np.int_)
        # (num_samples, 3)
        rays_origin_dr = dr.gather(
            self._mi_vec_t,
            self._mi_tensor_t(sources_positions).array,
            samples_tx_indices_dr,
        )
        rays_origin = mi_to_np_ndarray(rays_origin_dr, np.float_)
        # Rays
        ray = mi.Ray3f(o=rays_origin_dr, d=k_tx_dr)

        # Previous intersection point. Initialized to the transmitter position
        # (num_samples, 3)
        previous_int_point = rays_origin

        # Initializing the coverage map
        # Add dummy row and columns to store the items that are out of the
        # coverage map
        # (num_tx, num_cells_y+1, num_cells_x+1)
        cm = np.zeros([num_tx, num_cells_y + 1, num_cells_x + 1], dtype=np.float_)

        # Reflections on RIS change the radii and principal directions of
        # curvature of incident spherical waves.
        # Waves scattered by RIS are therefore not spherical even if the
        # incident waves are.
        # We need to therefore keep track of these quantities for every ray.
        # Radii of curvatures are initialized to 0
        # (num_samples, 2)
        radii_curv = np.zeros([num_samples, 2], dtype=np.float_)
        # Principal directions of curvatures are represented in the GCS.
        # Waves radiated by the transmitter are spherical, and therefore any
        # vectors u,v such that (u,v,k) is an orthonormal basis and where k is
        # the direction of propagation are principal directions of curvature.
        # (num_samples, 3)
        dir_curv_1, dir_curv_2 = gen_basis_from_z(k_tx, SolverBase.EPSILON)
        dirs_curv = np.stack([dir_curv_1, dir_curv_2], axis=1)
        # Angular opening of the ray tube
        # (num_samples,)
        angular_opening = np.full(
            [num_samples], (4.0 * np.pi / samples_per_tx_float).astype(np.float_)
        )

        # Offset to apply to the Mitsuba shape modeling RIS to get the
        # corresponding objects ids
        if len(self._scene.objects) > 0:
            ris_ind_offset = max(obj.object_id for obj in self._scene.objects.values())
        else:
            ris_ind_offset = 0
        # Because Mitsuba does not necessarily assign IDs starting from 1,
        # we need to account for this offset
        ris_mi_ids = mi_to_np_ndarray(
            dr.reinterpret_array_v(mi.UInt32, ris_scene.shapes_dr()), np.int_
        )
        ris_ind_offset -= np.min(ris_mi_ids) - 1

        for depth in range(max_depth + 1):

            ################################################
            # Intersection test
            ################################################

            # Intersect with scene
            si_scene = self._mi_scene.ray_intersect(ray)

            # Intersect with the measurement plane
            si_mp = meas_plane.ray_intersect(ray)

            # Intersect with RIS
            # It is required to split the kernel as intersections are
            # tested with another Mitsuba scene containing the RIS
            if ris:
                dr.eval(si_scene, si_mp)
                si_ris = ris_scene.ray_intersect(ray)
                dr.eval(si_ris)
                si_ris_t = si_ris.t
                si_ris_val = si_ris.is_valid()
            else:
                si_ris_t = np.inf
                si_ris_val = False

            hit_scene_dr = si_scene.is_valid() & (si_scene.t < si_ris_t)
            hit_ris_dr = si_ris_val & (si_ris_t <= si_scene.t)

            # A ray is active if it interacted with the scene or a RIS
            # (num_samples,)
            active_dr = hit_scene_dr | hit_ris_dr
            # (num_samples,)
            hit_scene = mi_to_np_ndarray(hit_scene_dr, np.bool_)
            hit_ris = mi_to_np_ndarray(hit_ris_dr, np.bool_)
            active = mi_to_np_ndarray(active_dr, np.bool_)

            # Hit the measurement plane?
            # An intersection with the coverage map is only valid if it was
            # not obstructed
            # (num_samples,)
            hit_mp_dr = si_mp.is_valid() & (si_mp.t < si_scene.t) & (si_mp.t < si_ris_t)
            # (num_samples,)
            hit_mp = mi_to_np_ndarray(hit_mp_dr, np.bool_)

            # Discard LoS if requested
            # (num_samples,)
            hit_mp &= los or (depth > 0)

            ################################################
            # Initialize the electric field
            ################################################

            # The field is initialized with the transmit field in the GCS
            # at the first iteration for rays that either hit the coverage map
            # or are active
            if depth == 0:
                init_ray_dr = active_dr | si_mp.is_valid()
                init_ray = mi_to_np_ndarray(init_ray_dr, np.bool_)
                e_field, field_es, field_ep = self._init_e_field(
                    init_ray, samples_tx_indices, k_tx, tx_rot_mat
                )

            ################################################
            # Update the coverage map
            ################################################
            # Intersection point with the measurement plane
            # (num_samples, 3)
            mp_hit_point = ray.o + si_mp.t * ray.d
            mp_hit_point = mi_to_np_ndarray(mp_hit_point, np.float_)

            cm = self._update_coverage_map(
                cm_center,
                cm_size,
                cm_cell_size,
                num_cells,
                rot_gcs_2_mp,
                cm_normal,
                tx_rot_mat,
                rx_rot_mat,
                precoding_vec,
                combining_vec,
                samples_tx_indices,
                e_field,
                field_es,
                field_ep,
                mp_hit_point,
                hit_mp,
                k_tx,
                previous_int_point,
                cm,
                radii_curv,
                angular_opening,
            )

            # If the maximum requested depth is reached, we stop, as we just
            # updated the coverage map with the last requested contribution from
            # the rays.
            # We also stop if there is no remaining active ray.
            if (depth == max_depth) or (not np.any(active)):
                break

            #############################################
            # Extract primitives and RIS that were hit by
            # active rays.
            #############################################

            # Extract the scene primitives that were hit
            if dr.shape(self._shape_indices)[0] > 0:  # Scene is not empty
                shape_i = dr.gather(
                    mi.Int32,
                    self._shape_indices,
                    dr.reinterpret_array_v(mi.UInt32, si_scene.shape),
                    hit_scene_dr,
                )
                offsets = dr.gather(mi.Int32, self._prim_offsets, shape_i, hit_scene_dr)
                scene_primitives = offsets + si_scene.prim_index
            else:  # Scene is empty
                scene_primitives = dr.zeros(mi.Int32, dr.shape(hit_scene_dr)[0])

            # Extract indices of RIS that were hit
            if ris:
                ris_ind = (
                    dr.reinterpret_array_v(mi.UInt32, si_ris.shape) + ris_ind_offset
                )
            else:
                ris_ind = dr.zeros(mi.Int32, dr.shape(hit_scene_dr)[0])

            # Combine into a single array
            # (num_samples,)
            primitives = dr.select(hit_scene_dr, scene_primitives, ris_ind)
            primitives = dr.select(active_dr, primitives, -1)
            primitives = mi_to_np_ndarray(primitives, np.int_)

            # If diffraction is enabled, stores the primitives in LoS
            # for sampling their wedges. These are needed to compute the
            # coverage map for diffraction (not in this function).
            if diffraction and (depth == 0):
                # (num_samples,)
                los_primitives = dr.select(hit_scene_dr, scene_primitives, -1)
                los_primitives = mi_to_np_ndarray(los_primitives, np.int_)

            # At this point, max_depth > 0 and there are still active rays.
            # However, we can stop if neither reflection, scattering or
            # reflection from RIS is enabled, as only these phenomena require to
            # go further.
            if not (reflection or scattering or ris):
                break

            #############################################
            # Update the field.
            # Only active rays are updated.
            #############################################

            # Intersection point
            # (num_samples, 3)
            int_point = dr.select(
                hit_scene_dr, ray.o + si_scene.t * ray.d, ray.o + si_ris_t * ray.d
            )
            int_point = mi_to_np_ndarray(int_point, np.float_)

            # Sample scattering/reflection phenomena.
            # reflect_ind : (num_reflected_samples,)
            #   Indices of the rays that are reflected
            #  scatter_ind : (num_scattered_samples,)
            #   Indices of the rays that are scattered
            reflect_ind, scatter_ind = self._sample_interaction_phenomena(
                hit_scene,
                int_point,
                primitives,
                scattering_coefficient,
                reflection,
                scattering,
            )

            # Indices of the rays that hit RIS
            # (num_ris_reflected_samples,)
            ris_reflect_ind = np.argwhere(hit_ris)[:, 0]
            updated_e_field = np.zeros([0, e_field.shape[1], 2], self._dtype)
            updated_field_es = np.zeros([0, 3], np.float_)
            updated_field_ep = np.zeros([0, 3], np.float_)
            updated_int_point = np.zeros([0, 3], np.float_)
            updated_k_r = np.zeros([0, 3], np.float_)
            updated_radii_curv = np.zeros([0, 2], np.float_)
            updated_dirs_curv = np.zeros([0, 2, 3], np.float_)
            updated_ang_opening = np.zeros([0], np.float_)

            if reflect_ind.shape[0] > 0:
                # ref_e_field : (num_reflected_samples, num_tx_patterns, 2)
                # ref_field_es : (num_reflected_samples, 3)
                # ref_field_ep : (num_reflected_samples, 3)
                # ref_int_point : (num_reflected_samples, 3)
                # ref_k_r : (num_reflected_samples, 3)
                # ref_radii_curv : (num_reflected_samples, 2)
                # ref_dirs_curv : (num_reflected_samples, 2, 3)
                # ref_ang_opening : (num_reflected_samples,)
                (
                    ref_e_field,
                    ref_field_es,
                    ref_field_ep,
                    ref_int_point,
                    ref_k_r,
                    ref_radii_curv,
                    ref_dirs_curv,
                    ref_ang_opening,
                ) = self._apply_reflection(
                    reflect_ind,
                    int_point,
                    previous_int_point,
                    primitives,
                    e_field,
                    field_es,
                    field_ep,
                    etas,
                    scattering_coefficient,
                    scattering,
                    radii_curv,
                    dirs_curv,
                    angular_opening,
                )

                updated_e_field = np.concatenate([updated_e_field, ref_e_field], axis=0)
                updated_field_es = np.concatenate(
                    [updated_field_es, ref_field_es], axis=0
                )
                updated_field_ep = np.concatenate(
                    [updated_field_ep, ref_field_ep], axis=0
                )
                updated_int_point = np.concatenate(
                    [updated_int_point, ref_int_point], axis=0
                )
                updated_k_r = np.concatenate([updated_k_r, ref_k_r], axis=0)
                updated_radii_curv = np.concatenate(
                    [updated_radii_curv, ref_radii_curv], axis=0
                )
                updated_dirs_curv = np.concatenate(
                    [updated_dirs_curv, ref_dirs_curv], axis=0
                )
                updated_ang_opening = np.concatenate(
                    [updated_ang_opening, ref_ang_opening], axis=0
                )

            if scatter_ind.shape[0] > 0:
                # scat_e_field : (num_scattered_samples, num_tx_patterns, 2)
                # scat_field_es : (num_scattered_samples, 3)
                # scat_field_ep : (num_scattered_samples, 3)
                # scat_int_point : (num_scattered_samples, 3)
                # scat_k_r : (num_scattered_samples, 3)
                # scat_radii_curv : (num_scattered_samples, 2)
                # scat_dirs_curv : (num_scattered_samples, 2, 3)
                # scat_ang_opening : (num_scattered_samples,)
                (
                    scat_e_field,
                    scat_field_es,
                    scat_field_ep,
                    scat_int_point,
                    scat_k_r,
                    scat_radii_curv,
                    scat_dirs_curv,
                    scat_ang_opening,
                ) = self._apply_scattering(
                    scatter_ind,
                    int_point,
                    previous_int_point,
                    primitives,
                    e_field,
                    field_es,
                    field_ep,
                    etas,
                    scattering_coefficient,
                    xpd_coefficient,
                    alpha_r,
                    alpha_i,
                    lambda_,
                    reflection,
                    radii_curv,
                    dirs_curv,
                    angular_opening,
                )

                updated_e_field = np.concatenate(
                    [updated_e_field, scat_e_field], axis=0
                )
                updated_field_es = np.concatenate(
                    [updated_field_es, scat_field_es], axis=0
                )
                updated_field_ep = np.concatenate(
                    [updated_field_ep, scat_field_ep], axis=0
                )
                updated_int_point = np.concatenate(
                    [updated_int_point, scat_int_point], axis=0
                )
                updated_k_r = np.concatenate([updated_k_r, scat_k_r], axis=0)
                updated_radii_curv = np.concatenate(
                    [updated_radii_curv, scat_radii_curv], axis=0
                )
                updated_dirs_curv = np.concatenate(
                    [updated_dirs_curv, scat_dirs_curv], axis=0
                )
                updated_ang_opening = np.concatenate(
                    [updated_ang_opening, scat_ang_opening], axis=0
                )

            if ris_reflect_ind.shape[0] > 0:
                # ris_e_field : (num_ris_reflected_samples, num_tx_patterns, 2)
                # ris_field_es : (num_ris_reflected_samples, 3)
                # ris_field_ep : (num_ris_reflected_samples, 3)
                # ris_int_point : (num_ris_reflected_samples, 3)
                # ris_k_r : (num_ris_reflected_samples, 3)
                # ris_radii_curv : (num_ris_reflected_samples, 2)
                # ris_dirs_curv : (num_ris_reflected_samples, 2, 3)
                # ris_ang_opening : (num_ris_reflected_samples,)
                (
                    ris_e_field,
                    ris_field_es,
                    ris_field_ep,
                    ris_int_point,
                    ris_k_r,
                    ris_radii_curv,
                    ris_dirs_curv,
                    ris_ang_opening,
                ) = self._apply_ris_reflection(
                    ris_reflect_ind,
                    int_point,
                    previous_int_point,
                    primitives,
                    e_field,
                    field_es,
                    field_ep,
                    radii_curv,
                    dirs_curv,
                    angular_opening,
                )
                updated_e_field = np.concatenate([updated_e_field, ris_e_field], axis=0)
                updated_field_es = np.concatenate(
                    [updated_field_es, ris_field_es], axis=0
                )
                updated_field_ep = np.concatenate(
                    [updated_field_ep, ris_field_ep], axis=0
                )
                updated_int_point = np.concatenate(
                    [updated_int_point, ris_int_point], axis=0
                )
                updated_k_r = np.concatenate([updated_k_r, ris_k_r], axis=0)
                updated_radii_curv = np.concatenate(
                    [updated_radii_curv, ris_radii_curv], axis=0
                )
                updated_dirs_curv = np.concatenate(
                    [updated_dirs_curv, ris_dirs_curv], axis=0
                )
                updated_ang_opening = np.concatenate(
                    [updated_ang_opening, ris_ang_opening], axis=0
                )

            e_field = updated_e_field
            field_es = updated_field_es
            field_ep = updated_field_ep
            k_r = updated_k_r
            int_point = updated_int_point
            radii_curv = updated_radii_curv
            dirs_curv = updated_dirs_curv
            angular_opening = updated_ang_opening

            # Only keep TX indices for active rays
            # (num_active_samples,)
            samples_tx_indices = np.bool_ean_mask(samples_tx_indices, active)

            ###############################################
            # Discard paths which path loss is below a
            # threshold
            ###############################################
            # (num_samples,)
            e_field_en = np.sum(np.abs(e_field) ** 2, axis=(1, 2))
            active = e_field_en > SolverCoverageMap.DISCARD_THRES
            if not np.any(active):
                break
            # (num_active_samples,)
            active_ind = np.argwhere(active)[:, 0]
            # (num_active_samples, ...)
            e_field = e_field[active_ind]
            field_es = field_es[active_ind]
            field_ep = field_ep[active_ind]
            k_r = k_r[active_ind]
            int_point = int_point[active_ind]
            radii_curv = radii_curv[active_ind]
            dirs_curv = dirs_curv[active_ind]
            samples_tx_indices = samples_tx_indices[active_ind]
            ###############################################
            # Reflect or scatter the current ray
            ###############################################

            # Spawn a new rays
            # (num_active_samples, 3)
            k_r_dr = self._mi_vec_t(k_r)
            rays_origin_dr = self._mi_vec_t(int_point)
            rays_origin_dr += SolverBase.EPSILON_OBSTRUCTION * k_r_dr
            ray = mi.Ray3f(o=rays_origin_dr, d=k_r_dr)
            # Update previous intersection point
            # (num_active_samples, 3)
            previous_int_point = int_point

        #################################################
        # Finalize the computation of the coverage map
        #################################################

        # Scaling factor
        cell_area = (cm_cell_size[0] * cm_cell_size[1]).astype(np.float_)
        cm_scaling = (self._scene.wavelength / (4.0 * np.pi)) ** 2 / cell_area
        cm_scaling = cm_scaling.astype(np.float_)

        # Dump the dummy line and row and apply the scaling factor
        # (num_tx, num_cells_y, num_cells_x)
        cm = cm_scaling * cm[:, :num_cells_y, :num_cells_x]

        # For diffraction, we need only primitives in LoS
        # (num_los_primitives)
        if los_primitives is not None:
            los_primitives, _ = np.unique(los_primitives)

        return cm, los_primitives

    def _discard_obstructing_wedges(self, candidate_wedges, sources_positions):
        r"""
        Discard wedges for which the source is "inside" the wedge

        Input
        ------
        candidate_wedges : [num_candidate_wedges], int
            Candidate wedges.
            Entries correspond to wedges indices.

        sources_positions : (num_tx, 3), np.float_
            Coordinates of the sources.

        Output
        -------
        diff_mask : (num_tx, num_candidate_wedges), np.bool_
            Mask set to False for invalid wedges

        diff_wedges_ind : (num_candidate_wedges,), np.int_
            Indices of the wedges that interacted with the diffracted paths
        """

        epsilon = SolverBase.EPSILON

        # (num_candidate_wedges, 3)
        origins = self._wedges_origin[candidate_wedges]

        # Expand to broadcast with sources/targets and 0/n faces
        # (1, num_candidate_wedges, 1, 3)
        origins = np.expand_dims(origins, axis=0)
        origins = np.expand_dims(origins, axis=2)

        # Normals
        # (num_candidate_wedges, 2, 3)
        # [:,0,:] : 0-face
        # [:,1,:] : n-face
        normals = self._wedges_normals[candidate_wedges]
        # Expand to broadcast with the sources or targets
        # (1, num_candidate_wedges, 2, 3)
        normals = np.expand_dims(normals, axis=0)

        # Expand to broadcast with candidate and 0/n faces wedges
        # (num_tx, 1, 1, 3)
        sources_positions = sources_positions.reshape(
            (sources_positions.shape[0], 1, 1, 3)
        )
        # Sources vectors
        # (num_tx, num_candidate_wedges, 1, 3)
        u_t = sources_positions - origins

        # (num_tx, num_candidate_wedges, 2)
        mask = dot(u_t, normals)
        mask = mask[np.full(mask.shape, epsilon)]
        # (num_tx, num_candidate_wedges)
        mask = np.any(mask, axis=2)

        # Discard wedges with no valid link
        # (num_candidate_wedges,)
        valid_wedges = np.argwhere(np.any(mask, axis=0))[:, 0]
        # (num_tx, num_candidate_wedges)
        mask = mask[:, valid_wedges]
        # (num_candidate_wedges,)
        diff_wedges_ind = candidate_wedges[valid_wedges]

        return mask, diff_wedges_ind

    def _sample_wedge_points(self, diff_mask, diff_wedges_ind, num_samples):
        r"""

        Samples equally spaced candidate diffraction points on the candidate
        wedges.

        The distance between two points is the cumulative length of the
        candidate wedges divided by ``num_samples``, i.e., the density of
        samples is the same for all wedges.

        The `num_samples` dimension of the output tensors is in general slighly
        smaller than input `num_samples` because of roundings.

        Input
        ------
        diff_mask : (num_tx, num_samples,), np.bool_
            Mask set to False for invalid samples

        diff_wedges_ind : [num_candidate_wedges], int
            Candidate wedges indices

        num_samples : int
            Number of samples to shoot

        Output
        ------
        diff_mask : (num_tx, num_samples), np.bool_
            Mask set to False for invalid wedges

        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : (num_samples,), np.float_
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_vertex : (num_samples, 3), np.float_
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : (num_samples,), np.int_
            For each sample, total mumber of samples that were sampled on the
            same wedge
        """

        zero_dot_five = 0.5

        # (num_candidate_wedges,)
        wedges_length = self._wedges_length[diff_wedges_ind]
        # Total length of the wedges
        # ()
        wedges_total_length = np.sum(wedges_length)
        # Spacing between the samples
        # ()
        delta_ell = wedges_total_length / float(num_samples)
        # Number of samples for each wedge
        # (num_candidate_wedges,)
        samples_per_wedge = np.where(delta_ell == 0.0, 0, wedges_length / delta_ell)
        samples_per_wedge = np.floor(samples_per_wedge).astype(np.int_)
        # Maximum number of samples for a wedge
        # np.maximum() required for the case where samples_per_wedge is empty
        # ()
        max_samples_per_wedge = np.maximum(np.max(samples_per_wedge), 0)
        # Sequence used to build the equally spaced samples on the wedges
        # (max_samples_per_wedge,)
        cseq = np.cumsum(np.ones([max_samples_per_wedge], dtype=np.int_)) - 1
        # (1, max_samples_per_wedge)
        cseq = np.expand_dims(cseq, axis=0)
        # (num_candidate_wedges, 1)
        samples_per_wedge_ = np.expand_dims(samples_per_wedge, axis=1)
        # (num_candidate_wedges, max_samples_per_wedge)
        ells_i = np.where(cseq < samples_per_wedge_, cseq, max_samples_per_wedge)
        # Compute the relative offset of the diffraction point on the wedge
        # (num_candidate_wedges, max_samples_per_wedge)
        ells = (ells_i.astype(np.float_) + zero_dot_five) * delta_ell
        # (num_candidate_wedges x max_samples_per_wedge)
        ells_i = np.reshape(ells_i, [-1])
        ells = np.reshape(ells, [-1])
        # Extract only relevant indices
        # (num_samples,). Smaller but close than input num_samples in general
        # because of previous floor() op
        # TODO-hermes: check if we can avoid np.argwhere in indexing like this
        ells = ells[np.argwhere(ells_i < max_samples_per_wedge)][:, 0]

        # Compute the corresponding points coordinates in the GCS
        # Wedges origin
        # (num_candidate_wedges, 3)
        origins = self._wedges_origin[diff_wedges_ind]
        # Wedges directions
        # (num_candidate_wedges, 3)
        e_hat = self._wedges_e_hat[diff_wedges_ind]
        # Match each sample to the corresponding wedge origin and vector
        # First, generate the indices for the gather op
        # ()
        num_candidate_wedges = diff_wedges_ind.shape[0]
        # (num_candidate_wedges,)
        gather_ind = np.arange(num_candidate_wedges)
        gather_ind = np.expand_dims(gather_ind, axis=1)
        # (num_candidate_wedges, max_samples_per_wedge)
        gather_ind = np.where(
            cseq < samples_per_wedge_, gather_ind, num_candidate_wedges
        )
        # (num_candidate_wedges x max_samples_per_wedge)
        gather_ind = gather_ind.reshape([-1])
        # (num_samples,)
        gather_ind = gather_ind[np.argwhere(ells_i < max_samples_per_wedge)][:, 0]
        # (num_samples, 3)
        origins = origins[gather_ind]
        e_hat = e_hat[gather_ind]
        # (num_samples,)
        diff_wedges_ind = diff_wedges_ind[gather_ind]
        # (num_tx, num_samples)
        diff_mask = diff_mask[:, gather_ind]
        # Positions of the diffracted points in the GCS
        # (num_samples, 3)
        diff_points = origins + np.expand_dims(ells, axis=1) * e_hat
        # Number of samples per wedge
        # (num_samples,)
        samples_per_wedge = samples_per_wedge[gather_ind]

        return diff_mask, diff_wedges_ind, ells, diff_points, samples_per_wedge

    def _test_tx_visibility(
        self,
        diff_mask,
        diff_wedges_ind,
        diff_ells,
        diff_vertex,
        diff_num_samples_per_wedge,
        sources_positions,
    ):
        r"""
        Test for blockage between the diffraction points and the transmitters.
        Blocked samples are discarded.

        Input
        ------
        diff_mask : (num_tx, num_samples), np.bool_
            Mask set to False for invalid samples

        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : (num_samples,), np.float_
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_vertex : (num_samples, 3), np.float_
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : (num_samples), np.int_
                For each sample, total mumber of samples that were sampled on
                the same wedge

        sources_positions : (num_tx, 3), np.float_
            Positions of the transmitters.

        Output
        -------
        diff_mask : (num_tx, num_samples), np.bool_
            Mask set to False for invalid wedges

        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : (num_samples,), np.float_
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_vertex : (num_samples, 3), np.float_
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : (num_samples,), np.int_
                For each sample, total mumber of samples that were sampled on
                the same wedge
        """

        num_tx = sources_positions.shape[0]
        num_samples = diff_vertex.shape[0]

        # (num_tx, 1, 3)
        sources_positions = np.expand_dims(sources_positions, axis=1)
        # (1, num_samples, 3)
        wedges_diff_points_ = np.expand_dims(diff_vertex, axis=0)
        # Ray directions and maximum distance for obstruction test
        # ray_dir : (num_tx, num_samples, 3)
        # maxt : (num_tx, num_samples)
        ray_dir, maxt = normalize(sources_positions - wedges_diff_points_)
        # Ray origins
        # (num_tx, num_samples, 3)
        ray_org = np.tile(wedges_diff_points_, [num_tx, 1, 1])

        # Test for obstruction
        # (num_tx, num_samples)
        ray_org = np.reshape(ray_org, [-1, 3])
        ray_dir = np.reshape(ray_dir, [-1, 3])
        maxt = np.reshape(maxt, [-1])
        invalid = self._test_obstruction(ray_org, ray_dir, maxt)
        invalid = np.reshape(invalid, [num_tx, num_samples])

        # Remove discarded paths
        # (num_tx, num_samples)
        diff_mask = np.logical_and(diff_mask, ~invalid)
        # Discard samples with no valid link
        # (num_candidate_wedges,)
        valid_samples = np.argwhere(np.any(diff_mask, axis=0))[:, 0]
        # (num_tx, num_samples)
        diff_mask = diff_mask[:, valid_samples]
        # (num_samples,)
        diff_wedges_ind = diff_wedges_ind[valid_samples]
        # (num_samples,)
        diff_vertex = diff_vertex[valid_samples]
        # (num_samples,)
        diff_ells = diff_ells[valid_samples]
        # (num_samples,)
        diff_num_samples_per_wedge = diff_num_samples_per_wedge[valid_samples]

        return (
            diff_mask,
            diff_wedges_ind,
            diff_ells,
            diff_vertex,
            diff_num_samples_per_wedge,
        )

    def _sample_diff_angles(self, diff_wedges_ind):
        r"""
        Samples angles of diffracted ray on the diffraction cone

        Input
        ------
        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        Output
        -------
        diff_phi : (num_samples,), np.float_
            Sampled angles of diffracted rays on the diffraction cone
        """

        num_samples = diff_wedges_ind.shape[0]

        # (num_samples, 2, 3)
        normals = self._wedges_normals[diff_wedges_ind]

        # Compute the wedges angle
        # (num_samples,)
        cos_wedges_angle = dot(normals[:, 0, :], normals[:, 1, :], clip=True)
        wedges_angle = np.pi + np.arccos(cos_wedges_angle)

        # Uniformly sample angles for shooting rays on the diffraction cone
        # (num_samples,)
        phis = np.random.uniform(
            [num_samples],
            low=np.zeros_like(wedges_angle),
            high=wedges_angle,
            dtype=np.float_,
        )

        return phis

    def _shoot_diffracted_rays(
        self,
        diff_mask,
        diff_wedges_ind,
        diff_ells,
        diff_vertex,
        diff_num_samples_per_wedge,
        diff_phi,
        sources_positions,
        meas_plane,
    ):
        r"""
        Shoots the diffracted rays and computes their intersection with the
        coverage map, if any. Rays blocked by the scene are discarded. Rays
        that do not hit the coverage map are discarded.

        Input
        ------
        diff_mask : (num_tx, num_samples), np.bool_
            Mask set to False for invalid samples

        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : (num_samples,), np.float_
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_vertex : (num_samples, 3), np.float_
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : (num_samples,), np.int_
            For each sample, total mumber of samples that were sampled on the
            same wedge

        diff_phi : (num_samples,), np.float_
            Sampled angles of diffracted rays on the diffraction cone

        sources_positions : (num_tx, 3), np.float_
            Positions of the transmitters.

        meas_plane : mi.Shape
            Mitsuba rectangle defining the measurement plane

        Output
        -------
        diff_mask : (num_tx, num_samples), np.bool_
            Mask set to False for invalid samples

        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : (num_samples,), np.float_
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins.

        diff_phi : (num_samples,), np.float_
            Sampled angles of diffracted rays on the diffraction cone

        diff_vertex : (num_samples, 3), np.float_
            Positions of the diffracted points in the GCS

        diff_num_samples_per_wedge : (num_samples,), np.int_
            For each sample, total mumber of samples that were sampled on the
            same wedge

        diff_hit_points : (num_tx, num_samples, 3), np.float_
            Positions of the intersection of the diffracted rays and coverage
            map

        diff_cone_angle : (num_tx, num_samples), np.float_
            Angle between e_hat and the diffracted ray direction.
            Takes value in (0,pi).
        """

        # (num_tx, 1, 3)
        sources_positions = np.expand_dims(sources_positions, axis=1)
        # (1, num_samples, 3)
        diff_points_ = np.expand_dims(diff_vertex, axis=0)
        # Ray directions and maximum distance for obstruction test
        # ray_dir : (num_tx, num_samples, 3)
        # maxt : (num_tx, num_samples)
        ray_dir, _ = normalize(diff_points_ - sources_positions)

        # Edge vector
        # (num_samples, 3)
        e_hat = self._wedges_e_hat[diff_wedges_ind]
        # (1, num_samples, 3)
        e_hat_ = np.expand_dims(e_hat, axis=0)
        # Angles between the incident ray and wedge.
        # This angle is not beta_0. It takes values in (0,pi), and is the angle
        # with respect to e_hat in which to shoot the diffracted ray.
        # (num_tx, num_samples)
        theta_shoot_dir = np.arccos(dot(ray_dir, e_hat_))

        # Discard paths for which the incident ray is aligned or perpendicular
        # to the edge
        # (num_tx, num_samples, 3)
        invalid_angle = np.stack(
            [
                theta_shoot_dir < SolverBase.EPSILON,
                theta_shoot_dir > np.pi - SolverBase.EPSILON,
                np.abs(theta_shoot_dir - 0.5 * np.pi) < SolverBase.EPSILON,
            ],
            axis=-1,
        )
        # (num_tx, num_samples)
        invalid_angle = np.any(invalid_angle, axis=-1)

        num_tx = diff_mask.shape[0]

        # Build the direction of the diffracted ray in the LCS
        # The LCS is defined by (t_0_hat, n0_hat, e_hat)

        # Direction of the diffracted ray
        # (1, num_samples)
        phis = np.expand_dims(diff_phi, axis=0)
        # (num_tx, num_samples, 3)
        diff_dir = r_hat(theta_shoot_dir, phis)

        # Matrix for going from the LCS to the GCS

        # Normals to face 0
        # (num_samples, 2, 3)
        normals = self._wedges_normals[diff_wedges_ind]
        # (num_samples, 3)
        normals = normals[:, 0, :]
        # Tangent vector t_hat
        # (num_samples, 3)
        t_hat = cross(normals, e_hat)
        # Matrix for going from LCS to GCS
        # (num_samples, 3, 3)
        lcs2gcs = np.stack([t_hat, normals, e_hat], axis=-1)
        # (1, num_samples, 3, 3)
        lcs2gcs = np.expand_dims(lcs2gcs, axis=0)

        # Direction of diffracted rays in CGS

        # (num_tx, num_samples, 3)
        diff_dir = matvec(lcs2gcs, diff_dir)

        # Origin of the diffracted rays

        # (num_tx, num_samples, 3)
        diff_points_ = np.tile(diff_points_, [num_tx, 1, 1])

        # Test of intersection of the diffracted rays with the measurement
        # plane
        mi_diff_dir = self._mi_vec_t(np.reshape(diff_dir, [-1, 3]))
        mi_diff_points = self._mi_vec_t(np.reshape(diff_points_, [-1, 3]))
        rays = mi.Ray3f(o=mi_diff_points, d=mi_diff_dir)
        # Intersect with the coverage map
        si_mp = meas_plane.ray_intersect(rays)

        # Check for obstruction
        # (num_tx x num_samples)
        obstructed = self._test_obstruction(mi_diff_points, mi_diff_dir, si_mp.t)

        # Mask invalid rays, i.e., rays that are obstructed or do that not hit
        # the measurement plane, and discard rays that are invalid for all TXs

        # (num_tx x num_samples)
        maxt = mi_to_np_ndarray(si_mp.t, dtype=np.float_)
        # (num_tx x num_samples)
        invalid = np.logical_or(np.isinf(maxt), obstructed)
        # (num_tx, num_samples)
        invalid = np.reshape(invalid, [num_tx, -1])
        # (num_tx, num_samples)
        invalid = np.logical_or(invalid, invalid_angle)
        # (num_tx, num_samples)
        diff_mask = np.logical_and(diff_mask, ~invalid)
        # Discard samples with no valid link
        # (num_candidate_wedges)
        valid_samples = np.argwhere(np.any(diff_mask, axis=0))[:, 0]
        # (num_tx, num_samples)
        diff_mask = diff_mask[:, valid_samples]
        # (num_samples,)
        diff_wedges_ind = diff_wedges_ind[valid_samples]
        # (num_samples,)
        diff_ells = diff_ells[valid_samples]
        # (num_samples,)
        diff_phi = diff_phi[valid_samples]
        # (num_tx, num_samples)
        theta_shoot_dir = theta_shoot_dir[:, valid_samples]
        # (num_samples,)
        diff_num_samples_per_wedge = diff_num_samples_per_wedge[valid_samples]

        # Compute intersection point with the coverage map
        # (num_tx, num_samples)
        maxt = np.reshape(maxt, [num_tx, -1])
        # (num_tx, num_samples)
        maxt = maxt[:, valid_samples]
        # Zeros invalid samples to avoid numeric issues
        # (num_tx, num_samples)
        maxt = np.where(diff_mask, maxt, np.zeros_like(maxt))
        # (num_tx, num_samples, 1)
        maxt = np.expand_dims(maxt, -1)
        # (num_tx, num_samples, 3)
        diff_dir = diff_dir[:, valid_samples]
        # (num_samples, 3)
        diff_vertex = diff_vertex[valid_samples]
        # (num_tx, num_samples, 3)
        diff_hit_points = np.expand_dims(diff_vertex, axis=0) + maxt * diff_dir

        return (
            diff_mask,
            diff_wedges_ind,
            diff_ells,
            diff_phi,
            diff_vertex,
            diff_num_samples_per_wedge,
            diff_hit_points,
            theta_shoot_dir,
        )

    def _compute_samples_weights(
        self,
        cm_center,
        cm_orientation,
        sources_positions,
        diff_wedges_ind,
        diff_ells,
        diff_phi,
        diff_cone_angle,
    ):
        r"""
        Computes the weights for averaging the field powers of the samples to
        compute the Monte Carlo estimate of the integral of the diffracted field
        power over the measurement plane.

        These weights are required as the measurement plane is parametrized by
        the angle on the diffraction cones (phi) and position on the wedges
        (ell).

        Input
        ------
        cm_center : (3,), np.float_
            Center of the coverage map

        cm_orientation : (3,), np.float_
            Orientation of the coverage map

        sources_positions : (num_tx, 3), np.float_
            Coordinates of the sources

        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        diff_ells : (num_samples,), np.float_
            Positions of the diffraction points on the wedges.
            These positions are given as an offset from the wedges origins

        diff_phi : (num_samples,), np.float_
            Sampled angles of diffracted rays on the diffraction cone

        diff_cone_angle : (num_tx, num_samples), np.float_
            Angle between e_hat and the diffracted ray direction.
            Takes value in (0,pi).

        Output
        ------
        diff_samples_weights : (num_tx, num_samples), np.float_
            Weights for averaging the field powers of the samples.
        """
        # (1, 1, 3)
        cm_center = cm_center.reshape((1, 1, 3))
        # (num_tx, 1, 3)
        sources_positions = np.expand_dims(sources_positions, axis=1)

        # Normal to the coverage map
        # (3)
        cmo_z = cm_orientation[0]
        cmo_y = cm_orientation[1]
        cmo_x = cm_orientation[2]
        cm_normal = np.stack(
            [
                np.cos(cmo_z) * np.sin(cmo_y) * np.cos(cmo_x)
                + np.sin(cmo_z) * np.sin(cmo_x),
                np.sin(cmo_z) * np.sin(cmo_y) * np.cos(cmo_x)
                - np.cos(cmo_z) * np.sin(cmo_x),
                np.cos(cmo_y) * np.cos(cmo_x),
            ],
            axis=0,
        )
        # (1, 1, 3)
        cm_normal = cm_normal.reshape((1, 1, 3))

        # Origins
        # (num_samples, 3)
        origins = self._wedges_origin[diff_wedges_ind]
        # (1, num_samples, 3)
        origins = np.expand_dims(origins, axis=0)

        # Distance of the wedge to the measurement plane
        # (num_tx, num_samples)
        wedge_cm_dist = dot(cm_center - origins, cm_normal)

        # Edges vectors
        # (num_samples, 3)
        e_hat = self._wedges_e_hat[diff_wedges_ind]

        # Normals to face 0
        # (num_samples, 2, 3)
        normals = self._wedges_normals[diff_wedges_ind]
        # (num_samples, 3)
        normals = normals[:, 0, :]
        # Tangent vector t_hat
        # (num_samples, 3)
        t_hat = cross(normals, e_hat)
        # Matrix for going from LCS to GCS
        # (num_samples, 3, 3)
        gcs2lcs = np.stack([t_hat, normals, e_hat], axis=-2)
        # (1, num_samples, 3, 3)
        gcs2lcs = np.expand_dims(gcs2lcs, axis=0)
        # Normal in LCS
        # (1, num_samples, 3)
        cm_normal = matvec(gcs2lcs, cm_normal)

        # Projections of the transmitters on the wedges
        # (1, num_samples, 3)
        e_hat = np.expand_dims(e_hat, axis=0)
        # (num_tx, num_samples)
        tx_proj_org_dist = dot(sources_positions - origins, e_hat)
        # (num_tx, num_samples, 1)
        tx_proj_org_dist_ = np.expand_dims(tx_proj_org_dist, axis=2)

        # Position of the sources projections on the wedges
        # (num_tx, num_samples, 3)
        tx_proj_pos = origins + tx_proj_org_dist_ * e_hat
        # Distance of transmitters to wedges
        # (num_tx, num_samples)
        tx_wedge_dist = np.linalg.norm(tx_proj_pos - sources_positions, axis=-1)

        # Building the derivatives of the parametrization of the intersection
        # of the diffraction cone and measurement plane
        # (1, num_samples)
        diff_phi = np.expand_dims(diff_phi, axis=0)
        # (1, num_samples)
        diff_ells = np.expand_dims(diff_ells, axis=0)

        # (1, num_samples)
        cos_phi = np.cos(diff_phi)
        # (1, num_samples)
        sin_phi = np.sin(diff_phi)
        # (1, num_samples)
        xy_dot = cm_normal[..., 0] * cos_phi + cm_normal[..., 1] * sin_phi
        # (num_tx, num_samples)
        ell_min_d = diff_ells - tx_proj_org_dist
        # (num_tx, num_samples)
        u = np.sign(ell_min_d)
        # (num_tx, num_samples)
        ell_min_d = np.abs(ell_min_d)
        # (num_tx, num_samples)
        s = np.where(
            diff_cone_angle < 0.5 * np.pi,
            np.ones_like(diff_cone_angle),
            -np.ones_like(diff_cone_angle),
        )
        # (num_tx, num_samples)
        q = s * tx_wedge_dist * xy_dot + cm_normal[..., 2] * ell_min_d
        q_square = np.square(q)
        inv_q = np.where(q == 0.0, 0.0, 1.0 / q)
        # (num_tx, num_samples)
        big_d_min_lz = wedge_cm_dist - diff_ells * cm_normal[..., 2]

        # (num_tx, num_samples, 3)
        v1 = np.stack(
            [
                s * big_d_min_lz * tx_wedge_dist * cos_phi,
                s * big_d_min_lz * tx_wedge_dist * sin_phi,
                wedge_cm_dist * ell_min_d + s * diff_ells * tx_wedge_dist * xy_dot,
            ],
            axis=-1,
        )
        # (num_tx, num_samples, 3)
        v2 = np.stack(
            [
                -s * cm_normal[..., 2] * tx_wedge_dist * cos_phi,
                -s * cm_normal[..., 2] * tx_wedge_dist * sin_phi,
                u * wedge_cm_dist + s * tx_wedge_dist * xy_dot,
            ],
            axis=-1,
        )
        # Derivative with respect to ell
        # (num_tx, num_samples, 3)
        ds_dl = (
            np.expand_dims(
                np.where(q_square == 0.0, 0.0, -u * cm_normal[..., 2] / q_square),
                axis=-1,
            )
            * v1
        )
        ds_dl = ds_dl + np.expand_dims(inv_q, axis=-1) * v2

        # Derivative with respect to phi
        # (num_tx, num_samples)
        w = -cm_normal[..., 0] * sin_phi + cm_normal[..., 1] * cos_phi
        # (num_tx, num_samples, 3)
        v3 = np.stack(
            [
                -s * big_d_min_lz * tx_wedge_dist * sin_phi,
                s * big_d_min_lz * tx_wedge_dist * cos_phi,
                s * diff_ells * tx_wedge_dist * w,
            ],
            axis=-1,
        )
        # (num_tx, num_samples, 3)
        ds_dphi = (
            np.expand_dims(
                np.where(q_square == 0.0, 0.0, -s * tx_wedge_dist * w / q_square),
                axis=-1,
            )
            * v1
        )
        ds_dphi = ds_dphi + np.expand_dims(inv_q, axis=-1) * v3

        # Weighting
        # (num_tx, num_samples)
        diff_samples_weights = np.linalg.norm(cross(ds_dl, ds_dphi), axis=-1)

        return diff_samples_weights

    def _compute_diffracted_path_power(
        self,
        sources_positions,
        sources_orientations,
        rx_orientation,
        combining_vec,
        precoding_vec,
        diff_mask,
        diff_wedges_ind,
        diff_vertex,
        diff_hit_points,
        relative_permittivity,
        scattering_coefficient,
    ):
        """
        Computes the power of the diffracted paths.

        Input
        ------
        sources_positions : (num_tx, 3), np.float_
            Positions of the transmitters.

        sources_orientations : (num_tx, 3), np.float_
            Orientations of the sources.

        rx_orientation : (3,), np.float_
            Orientation of the receiver.
            This is used to compute the antenna response and antenna pattern
            for an imaginary receiver located on the coverage map.

        combining_vec : (num_rx_ant,), np.complex_
            Combining vector.
            If set to `None`, then no combining is applied, and
            the energy received by all antennas is summed.

        precoding_vec : (num_tx or 1, num_tx_ant), np.complex_
            Precoding vectors of the transmitters

        diff_mask : (num_tx, num_samples), np.bool_
            Mask set to False for invalid samples

        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        diff_vertex : (num_samples, 3), np.float_
            Positions of the diffracted points in the GCS

        diff_hit_points : (num_tx, num_samples, 3), np.float_
            Positions of the intersection of the diffracted rays and coverage
            map

        relative_permittivity : (num_shape,), np.complex_
            Tensor containing the complex relative permittivity of all objects

        scattering_coefficient : (num_shape,), np.float_
            Tensor containing the scattering coefficients of all objects

        Output
        ------
        diff_samples_power : (num_tx, num_samples), np.float_
            Powers of the samples of diffracted rays.
        """

        def f(x):
            """F(x) Eq.(88) in [ITUR_P526]"""
            sqrt_x = np.sqrt(x)
            sqrt_pi_2 = np.sqrt(np.pi / 2.0)

            # Fresnel integral
            arg = sqrt_x / sqrt_pi_2
            s, c = fresnel(arg)
            f = s + 1.0j * c

            factor = sqrt_pi_2 * sqrt_x + 0.0j
            factor = factor * np.exp(1.0j * x)
            res = 1.0 + 1.0j - 2.0 * f

            return factor * res

        wavelength = self._scene.wavelength
        k = 2.0 * np.pi / wavelength

        # On CPU, indexing with -1 does not work. Hence we replace -1 by 0.
        # This makes no difference on the resulting paths as such paths
        # are not flaged as active.
        # (num_samples,)
        valid_wedges_idx = np.where(diff_wedges_ind == -1, 0, diff_wedges_ind)

        # (num_tx, 1, 3)
        sources_positions = np.expand_dims(sources_positions, axis=1)

        # Normals
        # (num_samples, 2, 3)
        normals = self._wedges_normals[valid_wedges_idx]

        # Compute the wedges angle
        # (num_samples,)
        cos_wedges_angle = dot(normals[..., 0, :], normals[..., 1, :], clip=True)
        wedges_angle = np.pi - np.arccos(cos_wedges_angle)
        n = (2.0 * np.pi - wedges_angle) / np.pi
        # (1, num_samples)
        n = np.expand_dims(n, axis=0)

        # (num_samples, 3)
        e_hat = self._wedges_e_hat[valid_wedges_idx]
        # (1, num_samples, 3)
        e_hat = np.expand_dims(e_hat, axis=0)

        # Extract surface normals
        # (num_samples, 3)
        n_0_hat = normals[:, 0, :]
        # (1, num_samples, 3)
        n_0_hat = np.expand_dims(n_0_hat, axis=0)
        # (num_samples, 3)
        n_n_hat = normals[:, 1, :]
        # (1, num_samples, 3)
        n_n_hat = np.expand_dims(n_n_hat, axis=0)

        # Relative permitivities
        # (num_samples, 2)
        objects_indices = self._wedges_objects[valid_wedges_idx]

        # Relative permitivities and scattering coefficients
        # If a callable is defined to compute the radio material properties,
        # it is invoked. Otherwise, the radio materials of objects are used.
        rm_callable = self._scene.radio_material_callable
        if rm_callable is None:
            # (num_samples, 2)
            etas = relative_permittivity[objects_indices]
            scattering_coefficient = scattering_coefficient[objects_indices]
        else:
            # Harmonize the shapes of the radio material callables
            # (num_samples, 2, 3)
            diff_vertex_ = np.tile(np.expand_dims(diff_vertex, axis=-2), [1, 2, 1])
            # scattering_coefficient, etas : [num_samples, 2]
            etas, scattering_coefficient, _ = rm_callable(objects_indices, diff_vertex_)

        # (num_samples,)
        eta_0 = etas[:, 0]
        eta_n = etas[:, 1]
        # (1, num_samples)
        eta_0 = np.expand_dims(eta_0, axis=0)
        eta_n = np.expand_dims(eta_n, axis=0)
        # (num_samples,)
        scattering_coefficient_0 = scattering_coefficient[..., 0]
        scattering_coefficient_n = scattering_coefficient[..., 1]
        # (1, num_samples)
        scattering_coefficient_0 = np.expand_dims(scattering_coefficient_0, axis=0)
        scattering_coefficient_n = np.expand_dims(scattering_coefficient_n, axis=0)

        # Compute s_prime_hat, s_hat, s_prime, s
        # (1, num_samples, 3)
        diff_vertex_ = np.expand_dims(diff_vertex, axis=0)
        # s_prime_hat : (num_tx, num_samples, 3)
        # s_prime : (num_tx, num_samples)
        s_prime_hat, s_prime = normalize(diff_vertex_ - sources_positions)
        # s_hat : (num_tx, num_samples, 3)
        # s : (num_tx, num_samples)
        s_hat, s = normalize(diff_hit_points - diff_vertex_)

        # Compute phi_prime_hat, beta_0_prime_hat, phi_hat, beta_0_hat
        # (num_tx, num_samples, 3)
        phi_prime_hat, _ = normalize(cross(s_prime_hat, e_hat))
        # (num_tx, num_samples, 3)
        beta_0_prime_hat = cross(phi_prime_hat, s_prime_hat)

        # (num_tx, num_samples, 3)
        phi_hat_, _ = normalize(-cross(s_hat, e_hat))
        beta_0_hat = cross(phi_hat_, s_hat)

        # Compute tangent vector t_0_hat
        # (1, num_samples, 3)
        t_0_hat = cross(n_0_hat, e_hat)

        # Compute s_t_prime_hat and s_t_hat
        # (num_tx, num_samples, 3)
        s_t_prime_hat, _ = normalize(
            s_prime_hat - dot(s_prime_hat, e_hat, keepdim=True) * e_hat
        )
        # (num_tx, num_samples, 3)
        s_t_hat, _ = normalize(s_hat - dot(s_hat, e_hat, keepdim=True) * e_hat)

        # Compute phi_prime and phi
        # (num_tx, num_samples)
        phi_prime = np.pi - (np.pi - np.arccos(-dot(s_t_prime_hat, t_0_hat))) * sign(
            -dot(s_t_prime_hat, n_0_hat)
        )
        # (num_tx, num_samples)
        phi = np.pi - (np.pi - np.arccos(dot(s_t_hat, t_0_hat))) * sign(
            dot(s_t_hat, n_0_hat)
        )

        # Compute field component vectors for reflections at both surfaces
        # (num_tx, num_samples, 3)
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_0, e_i_p_0, e_r_s_0, e_r_p_0 = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_0_hat,  # *sign(-dot(s_t_prime_hat, n_0_hat, keepdim=True)),
            SolverBase.EPSILON,
        )
        # (num_tx, num_samples, 3)
        # pylint: disable=unbalanced-tuple-unpacking
        e_i_s_n, e_i_p_n, e_r_s_n, e_r_p_n = compute_field_unit_vectors(
            s_prime_hat,
            s_hat,
            n_n_hat,  # *sign(-dot(s_t_prime_hat, n_n_hat, keepdim=True)),
            SolverBase.EPSILON,
        )

        # Compute Fresnel reflection coefficients for 0- and n-surfaces
        # (num_tx, num_samples)
        r_s_0, r_p_0 = reflection_coefficient(eta_0, np.abs(np.sin(phi_prime)))
        r_s_n, r_p_n = reflection_coefficient(eta_n, np.abs(np.sin(n * np.pi - phi)))

        # Multiply the reflection coefficients with the
        # corresponding reflection reduction factor
        reduction_factor_0 = np.sqrt(1 - scattering_coefficient_0**2)
        reduction_factor_0 = reduction_factor_0 + 0.0j
        reduction_factor_n = np.sqrt(1 - scattering_coefficient_n**2)
        reduction_factor_n = reduction_factor_n + 0.0j
        r_s_0 *= reduction_factor_0
        r_p_0 *= reduction_factor_0
        r_s_n *= reduction_factor_n
        r_p_n *= reduction_factor_n

        # Compute matrices R_0, R_n
        # (num_tx, num_samples, 2, 2)
        w_i_0 = component_transform(phi_prime_hat, beta_0_prime_hat, e_i_s_0, e_i_p_0)
        w_i_0 = w_i_0 + 0.0j
        # (num_tx, num_samples, 2, 2)
        w_r_0 = component_transform(e_r_s_0, e_r_p_0, phi_hat_, beta_0_hat)
        w_r_0 = w_r_0 + 0.0j
        # (num_tx, num_samples, 2, 2)
        r_0 = np.expand_dims(np.stack([r_s_0, r_p_0], -1), -1) * w_i_0
        # (num_tx, num_samples, 2, 2)
        r_0 = -(w_r_0 @ r_0)

        # (num_tx, num_samples, 2, 2)
        w_i_n = component_transform(phi_prime_hat, beta_0_prime_hat, e_i_s_n, e_i_p_n)
        w_i_n = w_i_n + 0.0j
        # (num_tx, num_samples, 2, 2)
        w_r_n = component_transform(e_r_s_n, e_r_p_n, phi_hat_, beta_0_hat)
        w_r_n = w_r_n + 0.0j
        # (num_tx, num_samples, 2, 2)
        r_n = np.expand_dims(np.stack([r_s_n, r_p_n], -1), -1) * w_i_n
        # (num_tx, num_samples, 2, 2)
        r_n = -(w_r_n @ r_n)

        # Compute D_1, D_2, D_3, D_4
        # (num_tx, num_samples)
        phi_m = phi - phi_prime
        phi_p = phi + phi_prime

        # (num_tx, num_samples)
        cot_1 = cot((np.pi + phi_m) / (2 * n))
        cot_2 = cot((np.pi - phi_m) / (2 * n))
        cot_3 = cot((np.pi + phi_p) / (2 * n))
        cot_4 = cot((np.pi - phi_p) / (2 * n))

        def n_p(beta, n):
            return np.round((beta + np.pi) / (2.0 * n * np.pi))

        def n_m(beta, n):
            return np.round((beta - np.pi) / (2.0 * n * np.pi))

        def a_p(beta, n):
            return 2 * np.cos((2.0 * n * np.pi * n_p(beta, n) - beta) / 2.0) ** 2

        def a_m(beta, n):
            return 2 * np.cos((2.0 * n * np.pi * n_m(beta, n) - beta) / 2.0) ** 2

        # (1, num_samples)
        d_mul = -np.exp(-1j * np.pi / 4.0) / ((2 * n) * np.sqrt(2 * np.pi * k))

        # (num_tx, num_samples)
        ell = s_prime * s / (s_prime + s)

        # (num_tx, num_samples)
        cot_1 = cot_1 + 0.0j
        cot_2 = cot_2 + 0.0j
        cot_3 = cot_3 + 0.0j
        cot_4 = cot_4 + 0.0j
        d_1 = d_mul * cot_1 * f(k * ell * a_p(phi_m, n))
        d_2 = d_mul * cot_2 * f(k * ell * a_m(phi_m, n))
        d_3 = d_mul * cot_3 * f(k * ell * a_p(phi_p, n))
        d_4 = d_mul * cot_4 * f(k * ell * a_m(phi_p, n))

        # (num_tx, num_samples, 1, 1)
        d_1 = np.reshape(d_1, np.concatenate([d_1.shape, [1, 1]], axis=0))
        d_2 = np.reshape(d_2, np.concatenate([d_2.shape, [1, 1]], axis=0))
        d_3 = np.reshape(d_3, np.concatenate([d_3.shape, [1, 1]], axis=0))
        d_4 = np.reshape(d_4, np.concatenate([d_4.shape, [1, 1]], axis=0))

        # (num_tx, num_samples)
        divisor = s * s_prime * (s_prime + s)
        spreading_factor = np.where(divisor == 0.0, 0.0, 1.0 / divisor)
        spreading_factor = np.sqrt(spreading_factor)
        spreading_factor = spreading_factor + 0.0j
        # (num_tx, num_samples, 1, 1)
        spreading_factor = np.reshape(spreading_factor, d_1.shape)

        # (num_tx, num_samples, 2, 2)
        mat_t = (d_1 + d_2) * np.eye(2, 2, batch_shape=r_0.shape[:2], dtype=self._dtype)
        # (num_tx, num_samples, 2, 2)
        mat_t += d_3 * r_n + d_4 * r_0
        # (num_tx, num_samples, 2, 2)
        mat_t *= -spreading_factor

        # Convert from/to GCS
        # (num_tx, num_samples)
        theta_t, phi_t = theta_phi_from_unit_vec(s_prime_hat)
        theta_r, phi_r = theta_phi_from_unit_vec(-s_hat)

        # (num_tx, num_samples, 2, 2)
        mat_from_gcs = component_transform(
            theta_hat(theta_t, phi_t), phi_hat(phi_t), phi_prime_hat, beta_0_prime_hat
        )
        mat_from_gcs = mat_from_gcs + 0.0j

        # (num_tx, num_samples, 2, 2)
        mat_to_gcs = component_transform(
            phi_hat_, beta_0_hat, theta_hat(theta_r, phi_r), phi_hat(phi_r)
        )
        mat_to_gcs = mat_to_gcs + 0.0j

        # (num_tx, num_samples, 2, 2)
        mat_t = mat_t @ mat_from_gcs
        mat_t = mat_to_gcs @ mat_t

        # Set invalid paths to 0
        # Expand masks to broadcast with the field components
        # (num_tx, num_samples, 1, 1)
        mask_ = diff_mask[..., np.newaxis, np.newaxis]
        # Zeroing coefficients corresponding to non-valid paths
        # (num_tx, num_samples, 2, 2)
        mat_t = np.where(mask_, mat_t, np.zeros_like(mat_t))

        # Compute transmitters antenna pattern in the GCS
        # (num_tx, 3, 3)
        tx_rot_mat = rotation_matrix(sources_orientations)
        # (num_tx, 1, 3, 3)
        tx_rot_mat = np.expand_dims(tx_rot_mat, axis=1)
        # tx_field : (num_tx, num_samples, num_tx_patterns, 2)
        # tx_es, ex_ep : (num_tx, num_samples, 3)
        tx_field, _, _ = self._compute_antenna_patterns(
            tx_rot_mat, self._scene.tx_array.antenna.patterns, s_prime_hat
        )

        # Compute receiver antenna pattern in the GCS
        # (3, 3)
        rx_rot_mat = rotation_matrix(rx_orientation)
        # tx_field : (num_tx, num_samples, num_rx_patterns, 2)
        # tx_es, ex_ep : (num_tx, num_samples, 3)
        rx_field, _, _ = self._compute_antenna_patterns(
            rx_rot_mat, self._scene.rx_array.antenna.patterns, -s_hat
        )

        # Compute the channel coefficients for every transmitter-receiver
        # pattern pairs
        # (num_tx, num_samples, 1, 1, 2, 2)
        mat_t = mat_t.reshape((mat_t.shape[0], mat_t.shape[1], 1, 1, 2, 2))
        # (num_tx, num_samples, 1, num_tx_patterns, 1, 2)
        tx_field = np.expand_dims(np.expand_dims(tx_field, axis=2), axis=4)
        # (num_tx, num_samples, num_rx_patterns, 1, 2)
        rx_field = np.expand_dims(rx_field, axis=3)
        # (num_tx, num_samples, 1, num_tx_patterns, 2)
        a = np.sum(mat_t * tx_field, axis=-1)
        # (num_tx, num_samples, num_rx_patterns, num_tx_patterns)
        a = np.sum(np.conj(rx_field) * a, axis=-1)

        # Apply synthetic array
        # (num_tx, num_samples, num_rx_antenna, num_tx_antenna)
        a = self._apply_synthetic_array(tx_rot_mat, rx_rot_mat, -s_hat, s_prime_hat, a)

        # Apply precoding
        # Precoding and combing
        # (num_tx/1, 1, 1, num_tx_ant)
        precoding_vec = precoding_vec.reshape(
            (precoding_vec.shape[1], 1, 1, precoding_vec.shape[-1])
        )
        # (num_tx, samples_per_tx, num_rx_ant)
        a = np.sum(a * precoding_vec, axis=-1)
        # Apply combining
        # If no combining vector is set, then the energy of all antennas is
        # summed
        if combining_vec is None:
            # (num_tx, samples_per_tx)
            a = np.sum(np.square(np.abs(a)), axis=-1)
        else:
            # (1, 1, num_rx_ant)
            combining_vec = combining_vec.reshape((1, 1, combining_vec.shape[-1]))
            # (num_tx, samples_per_tx)
            a = np.sum(np.conj(combining_vec) * a, axis=-1)
            # (num_tx, samples_per_tx)
            a = np.square(np.abs(a))

        # (num_tx, samples_per_tx)
        cst = np.square(self._scene.wavelength / (4.0 * np.pi))
        a = a * cst

        return a

    def _build_diff_coverage_map(
        self,
        cm_center,
        cm_orientation,
        cm_size,
        cm_cell_size,
        diff_wedges_ind,
        diff_hit_points,
        diff_samples_power,
        diff_samples_weights,
        diff_num_samples_per_wedge,
    ):
        r"""
        Builds the coverage map for diffraction

        Input
        ------
        cm_center : (3,), np.float_
            Center of the coverage map

        cm_orientation : (3,), np.float_
            Orientation of the coverage map

        cm_size : (2,), np.float_
            Scale of the coverage map.
            The width of the map (in the local X direction) is ``cm_size[0]``
            and its map (in the local Y direction) ``cm_size[1]``.

        cm_cell_size : (2,), np.float_
            Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            meters of a cell of the coverage map

        diff_wedges_ind : (num_samples,), np.int_
            Indices of the wedges that interacted with the diffracted paths

        diff_hit_points : (num_tx, num_samples, 3), np.float_
            Positions of the intersection of the diffracted rays and coverage
            map

        diff_samples_power : (num_tx, num_samples), np.float_
            Powers of the samples of diffracted rays.

        diff_samples_weights : (num_tx, num_samples), np.float_
            Weights for averaging the field powers of the samples.

        diff_num_samples_per_wedge : (num_samples,), np.int_
            For each sample, total mumber of samples that were sampled on the
            same wedge

        Output
        ------
        :cm : :class:`~sionna.rt.CoverageMap`
            The coverage maps
        """
        num_tx = diff_hit_points.shape[0]
        num_samples = diff_hit_points.shape[1]
        cell_area = cm_cell_size[0] * cm_cell_size[1]

        # (num_tx, num_samples)
        diff_wedges_ind = np.tile(np.expand_dims(diff_wedges_ind, axis=0), [num_tx, 1])

        # Transformation matrix required for computing the cell
        # indices of the intersection points
        # (3,3)
        rot_cm_2_gcs = rotation_matrix(cm_orientation)
        # (3,3)
        rot_gcs_2_cm = rot_cm_2_gcs.T

        # Initializing the coverage map
        num_cells_x = int(np.ceil(cm_size[0] / cm_cell_size[0]))
        num_cells_y = int(np.ceil(cm_size[1] / cm_cell_size[1]))
        num_cells = np.stack([num_cells_x, num_cells_y], axis=-1)
        # (num_tx, num_cells_y+1, num_cells_x+1)
        # Add dummy row and columns to store the items that are out of the
        # coverage map
        cm = np.zeros([num_tx, num_cells_y + 1, num_cells_x + 1], dtype=np.float_)

        # Coverage map cells' indices
        # (num_tx, num_samples, 2 : xy)
        cell_ind = self._mp_hit_point_2_cell_ind(
            rot_gcs_2_cm, cm_center, cm_size, cm_cell_size, num_cells, diff_hit_points
        )
        # Add the transmitter index to the coverage map
        # (num_tx)
        tx_ind = np.arange(num_tx, dtype=np.int_)
        # (num_tx, 1, 1)
        tx_ind = tx_ind[..., np.newaxis, np.newaxis]
        # (num_tx, num_samples, 1)
        tx_ind = np.tile(tx_ind, [1, num_samples, 1])
        # (num_tx, num_samples, 3)
        cm_ind = np.concatenate([tx_ind, cell_ind], axis=-1)

        # Wedges lengths
        # (num_tx, num_samples)
        lengths = self._wedges_length[diff_wedges_ind]

        # Wedges opening angles
        # (num_tx, num_samples, 2, 3)
        normals = self._wedges_normals[diff_wedges_ind]
        # (num_tx, num_samples)
        cos_op_angle = dot(normals[..., 0, :], normals[..., 1, :], clip=True)
        op_angles = np.pi + np.arccos(cos_op_angle)

        # Update the weights of each ray power
        # (1, num_samples)
        diff_num_samples_per_wedge = np.expand_dims(diff_num_samples_per_wedge, axis=0)
        diff_num_samples_per_wedge = float(diff_num_samples_per_wedge)
        # (num_tx, num_samples)
        diff_samples_weights = np.where(
            diff_num_samples_per_wedge == 0.0,
            0.0,
            diff_samples_weights / diff_num_samples_per_wedge,
        )
        diff_samples_weights = diff_samples_weights * lengths * op_angles

        # Add the weighted powers to the coverage map
        # (num_tx, num_samples)
        weighted_sample_power = diff_samples_power * diff_samples_weights
        # (num_tx, num_cells_y+1, num_cells_x+1)
        cm[cm_ind] = weighted_sample_power

        # Dump the dummy line and row
        # (num_tx, num_cells_y, num_cells_x)
        cm = cm[:, :num_cells_y, :num_cells_x]

        # Scaling by area of a cell
        # (num_tx, num_cells_y, num_cells_x)
        cm = cm / cell_area

        return cm

    def _diff_samples_2_coverage_map(
        self,
        los_primitives,
        edge_diffraction,
        num_samples,
        sources_positions,
        meas_plane,
        cm_center,
        cm_orientation,
        cm_size,
        cm_cell_size,
        sources_orientations,
        rx_orientation,
        combining_vec,
        precoding_vec,
        etas,
        scattering_coefficient,
    ):
        r"""
        Computes the coverage map for diffraction.

        Input
        ------
        los_primitives: [num_los_primitives], int
            Primitives in LoS.

        edge_diffraction : bool
            If set to `False`, only diffraction on wedges, i.e., edges that
            connect two primitives, is considered.

        num_samples : int
            Number of rays initially shooted from the wedges.

        sources_positions : (num_tx, 3), np.float_
            Coordinates of the sources.

        meas_plane : mi.Shape
            Mitsuba rectangle defining the measurement plane

        cm_center : (3,), np.float_
            Center of the coverage map

        cm_orientation : (3,), np.float_
            Orientation of the coverage map

        cm_size : (2,), np.float_
            Scale of the coverage map.
            The width of the map (in the local X direction) is ``cm_size[0]``
            and its map (in the local Y direction) ``cm_size[1]``.

        cm_cell_size : (2,), np.float_
            Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            meters of a cell of the coverage map

        sources_orientations : (num_tx, 3), np.float_
            Orientations of the sources.

        rx_orientation : (3,), np.float_
            Orientation of the receiver.

        combining_vec : (num_rx_ant,), np.complex_
            Combining vector.
            If set to `None`, then no combining is applied, and
            the energy received by all antennas is summed.

        precoding_vec : (num_tx or 1, num_tx_ant), np.complex_
            Precoding vectors of the transmitters

        etas : (num_shape,), np.complex_
            Tensor containing the complex relative permittivities of all shapes

        scattering_coefficient : (num_shape,), np.float_
            Tensor containing the scattering coefficients of all shapes

        Output
        -------
        :cm : :class:`~sionna.rt.CoverageMap`
            The coverage maps
        """

        # Build empty coverage map
        num_cells_x = int(np.ceil(cm_size[0] / cm_cell_size[0]))
        num_cells_y = int(np.ceil(cm_size[1] / cm_cell_size[1]))
        # (num_tx, num_cells_y, num_cells_x)
        cm_null = np.zeros(
            [sources_positions.shape[0], num_cells_y, num_cells_x], dtype=np.float_
        )

        # Get the candidate wedges for diffraction
        # diff_wedges_ind : (num_candidate_wedges), int
        #     Candidate wedges indices
        diff_wedges_ind = self._wedges_from_primitives(los_primitives, edge_diffraction)
        # Early stop if there are no wedges
        if diff_wedges_ind.shape[0] == 0:
            return cm_null

        # Discard wedges for which the tx is inside the wedge
        # diff_mask : (num_tx, num_candidate_wedges), bool
        #   Mask set to False if the wedge is invalid
        # wedges : (num_candidate_wedges), int
        #     Candidate wedges indices
        output = self._discard_obstructing_wedges(diff_wedges_ind, sources_positions)
        diff_mask = output[0]
        diff_wedges_ind = output[1]
        # Early stop if there are no wedges
        if diff_wedges_ind.shape[0] == 0:
            return cm_null

        # Sample diffraction points on the wedges
        # diff_mask : (num_tx, num_candidate_wedges), bool
        #   Mask set to False if the wedge is invalid
        # diff_wedges_ind : (num_candidate_wedges), int
        #     Candidate wedges indices
        # diff_ells : (num_samples,), float
        #   Positions of the diffraction points on the wedges.
        #   These positionsare given as an offset from the wedges origins.
        #   The size of this tensor is in general slighly smaller than
        #   `num_samples` because of roundings.
        # diff_vertex : (num_samples, 3), np.float_
        #   Positions of the diffracted points in the GCS
        # diff_num_samples_per_wedge : (num_samples,), np.int_
        #         For each sample, total mumber of samples that were sampled
        #         on the same wedge
        output = self._sample_wedge_points(diff_mask, diff_wedges_ind, num_samples)
        diff_mask = output[0]
        diff_wedges_ind = output[1]
        diff_ells = output[2]
        diff_vertex = output[3]
        diff_num_samples_per_wedge = output[4]

        # Test for blockage between the transmitters and diffraction points.
        # Discarted blocked samples.
        # diff_mask : (num_tx, num_candidate_wedges), bool
        #   Mask set to False if the wedge is invalid
        # diff_wedges_ind : (num_samples,), int
        #     Candidate wedges indices
        # diff_ells : (num_samples,), float
        #   Positions of the diffraction points on the wedges.
        #   These positionsare given as an offset from the wedges origins.
        #   The size of this tensor is in general slighly smaller than
        #   `num_samples` because of roundings.
        # diff_vertex : (num_samples, 3), float
        #   Positions of the diffracted points in the GCS
        # diff_num_samples_per_wedge : (num_samples,), np.int_
        #         For each sample, total mumber of samples that were sampled
        #         on the same wedge
        output = self._test_tx_visibility(
            diff_mask,
            diff_wedges_ind,
            diff_ells,
            diff_vertex,
            diff_num_samples_per_wedge,
            sources_positions,
        )
        diff_mask = output[0]
        diff_wedges_ind = output[1]
        diff_ells = output[2]
        diff_vertex = output[3]
        diff_num_samples_per_wedge = output[4]
        # Early stop if there are no wedges
        if diff_wedges_ind.shape[0] == 0:
            return cm_null

        # Samples angles for departure on the diffraction cone
        # diff_phi : (num_samples, 3), np.float_
        #   Sampled angles on the diffraction cone used for shooting rays
        diff_phi = self._sample_diff_angles(diff_wedges_ind)

        # Shoot rays in the sampled directions and test for intersection
        # with the coverage map.
        # Discard rays that miss it.
        # diff_mask : (num_tx, num_samples), np.bool_
        #     Mask set to False for invalid samples
        # diff_wedges_ind : (num_samples,), np.int_
        #     Indices of the wedges that interacted with the diffracted
        #     paths
        # diff_ells : (num_samples,), np.float_
        #     Positions of the diffraction points on the wedges.
        #     These positions are given as an offset from the wedges
        #     origins.
        # diff_phi : (num_samples,), np.float_
        #     Sampled angles of diffracted rays on the diffraction cone
        # diff_vertex : (num_samples, 3), np.float_
        #     Positions of the diffracted points in the GCS
        # diff_num_samples_per_wedge : (num_samples,), np.int_
        #         For each sample, total mumber of samples that were sampled
        #         on the same wedge
        # diff_hit_points : (num_tx, num_samples, 3), np.float_
        #     Positions of the intersection of the diffracted rays and
        #     coverage map
        # diff_cone_angle : (num_tx, num_samples), np.float_
        #     Angle between e_hat and the diffracted ray direction.
        #     Takes value in (0,pi).
        output = self._shoot_diffracted_rays(
            diff_mask,
            diff_wedges_ind,
            diff_ells,
            diff_vertex,
            diff_num_samples_per_wedge,
            diff_phi,
            sources_positions,
            meas_plane,
        )
        diff_mask = output[0]
        diff_wedges_ind = output[1]
        diff_ells = output[2]
        diff_phi = output[3]
        diff_vertex = output[4]
        diff_num_samples_per_wedge = output[5]
        diff_hit_points = output[6]
        diff_cone_angle = output[7]

        # Computes the weights for averaging the field powers of the samples
        # to compute the Monte Carlo estimate of the integral of the
        # diffracted field power over the measurement plane.
        # These weights are required as the measurement plane is
        # parametrized by the angle on the diffraction cones (phi) and
        # position on the wedges (ell).
        #
        # diff_samples_weights : (num_tx, num_samples), np.float_
        #     Weights for averaging the field powers of the samples.
        output = self._compute_samples_weights(
            cm_center,
            cm_orientation,
            sources_positions,
            diff_wedges_ind,
            diff_ells,
            diff_phi,
            diff_cone_angle,
        )
        diff_samples_weights = output

        # Computes the power of the diffracted paths.
        #
        # diff_samples_power : (num_tx, num_samples), np.float_
        #   Powers of the samples of diffracted rays.
        output = self._compute_diffracted_path_power(
            sources_positions,
            sources_orientations,
            rx_orientation,
            combining_vec,
            precoding_vec,
            diff_mask,
            diff_wedges_ind,
            diff_vertex,
            diff_hit_points,
            etas,
            scattering_coefficient,
        )
        diff_samples_power = output

        # Builds the coverage map for the diffracted field
        cm_diff = self._build_diff_coverage_map(
            cm_center,
            cm_orientation,
            cm_size,
            cm_cell_size,
            diff_wedges_ind,
            diff_hit_points,
            diff_samples_power,
            diff_samples_weights,
            diff_num_samples_per_wedge,
        )

        return cm_diff
