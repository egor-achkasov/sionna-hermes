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

from .utils import (
    normalize,
    dot,
    theta_phi_from_unit_vec,
    cross,
    mi_to_np_ndarray,
    mitsuba_rectangle_to_world,
)


class SolverBase:
    # pylint: disable=line-too-long
    r"""SolverBase(scene, solver=None, dtype=np.complex_)

    Base class for implementing a solver. If another ``solver`` is specified at
    instantiation, then it re-uses the structure to avoid useless compute and
    memory use.

    Note: Only triangle mesh are supported.

    Parameters
    -----------
    scene : :class:`~sionna.rt.Scene`
        Sionna RT scene

    solver : :class:`~sionna.rt.SolverBase` | None
        Another solver from which to re-use some structures to avoid useless
        compute and memory use

    dtype : np.dtype
        Datatype for all computations, inputs, and outputs.
        Defaults to `np.complex_`.
    """

    # Small value used to discard intersection with edges, avoid
    # self-intersection, etc.
    # Resolution of float32 1e-6:
    # np.finfo(np.float32) -> reslution=1e-6
    # We add on top a 10x factor for caution
    EPSILON = 1e-5

    # Threshold for extracting wedges from the scene [rad]
    WEDGES_ANGLE_THRESHOLD = 1.0 * np.pi / 180.0

    # Small value used to avoid false positive when testing for obstruction
    # Resolution of float32 1e-6:
    # np.finfo(np.float32) -> reslution=1e-6
    # We add on top a 100x factor for caution
    EPSILON_OBSTRUCTION = 1e-4

    def __init__(self, scene, solver=None, dtype=np.complex_):

        # Computes the quantities required for generating the paths.
        # More pricisely:

        # _primitives : [num triangles, 3, 3], float
        #     The triangles: x-y-z coordinates of the 3 vertices for every
        #     triangle

        # _normals : [num triangles, 3], float
        #     The normals of the triangles

        # _primitives_2_objects : [num_triangles], int
        #     Index of the shape containing the triangle

        # _prim_offsets : [num_objects], int
        #     Indices offsets for accessing the triangles making each shape.

        # _shape_indices : [num_objects], int
        #     Map Mitsuba shape indices to indices that can be used to access
        #    _prim_offsets

        self._dtype = dtype
        self._rdtype = np.float_

        # Mitsuba types depend on the used precision
        if dtype == np.complex_:
            self._mi_point_t = mi.Point3f
            self._mi_point2_t = mi.Point2f
            self._mi_vec_t = mi.Vector3f
            self._mi_scalar_t = mi.Float
            self._mi_tensor_t = mi.TensorXf
        else:
            self._mi_point_t = mi.Point3d
            self._mi_point2_t = mi.Point2d
            self._mi_vec_t = mi.Vector3d
            self._mi_scalar_t = mi.Float64
            self._mi_tensor_t = mi.TensorXd

        self._scene = scene
        mi_scene = scene.mi_scene
        self._mi_scene = mi_scene

        # If a solver is provided, then link to the same structures to avoid
        # useless compute and memory use
        if solver is not None:
            self._primitives = solver._primitives
            self._normals = solver._normals
            self._primitives_2_objects = solver._primitives_2_objects
            self._prim_offsets = solver._prim_offsets
            self._shape_indices = solver._shape_indices
            #
            self._wedges_origin = solver._wedges_origin
            self._wedges_e_hat = solver._wedges_e_hat
            self._wedges_length = solver._wedges_length
            self._wedges_normals = solver._wedges_normals
            self._primitives_2_wedges = solver._primitives_2_wedges
            self._wedges_objects = solver._wedges_objects
            self._is_edge = solver._is_edge
            return

        ###################################################
        # Extract triangles, their normals, and a
        # look-up-table to map primitives to the scene
        # object they belong to.
        ###################################################

        # Tensor mapping primitives to corresponding objects
        # [num_triangles]
        primitives_2_objects = []

        # Number of triangles
        n_prims = 0
        # Triangles of each object (shape) in the scene are stacked.
        # This list tracks the indices offsets for accessing the triangles
        # making each shape.
        prim_offsets = []
        objects_id = dr.reinterpret_array_v(mi.UInt32, mi_scene.shapes_dr())
        for i, s in zip(objects_id, mi_scene.shapes()):
            if not isinstance(s, mi.Mesh):
                raise ValueError("Only triangle meshes are supported")
            prim_offsets.append(n_prims)
            n_prims += s.face_count()
            primitives_2_objects += [i] * s.face_count()
        # [num_objects]
        prim_offsets = np.asarray(prim_offsets, np.int_)

        # Tensor of triangles vertices
        # [n_prims, number of vertices : 3, coordinates : 3]
        prims = np.zeros([n_prims, 3, 3], self._rdtype)
        # Normals to the triangles
        normals = np.zeros([n_prims, 3], self._rdtype)
        # Loop through the objects in the scene
        for prim_offset, s in zip(prim_offsets, mi_scene.shapes()):
            # Extract the vertices of the shape.
            # Dr.JIT/Mitsuba is used here.
            # Indices of the vertices
            # [n_prims, num of vertices per triangle : 3]
            face_indices3 = s.face_indices(dr.arange(mi.UInt32, s.face_count()))
            # Flatten. This is required for calling vertex_position
            # [n_prims*3]
            face_indices = dr.ravel(face_indices3)
            # Get vertices coordinates
            # [n_prims*3, 3]
            vertex_coords = s.vertex_position(face_indices)
            # Move to TensorFlow
            # [n_prims*3, 3]
            vertex_coords = mi_to_np_ndarray(vertex_coords, self._rdtype)
            # Unflatten
            # [n_prims, vertices per triangle : 3, 3]
            vertex_coords = np.reshape(vertex_coords, [s.face_count(), 3, 3])
            # Update the `prims` tensor
            sl = np.arange(prim_offset, prim_offset + s.face_count(), dtype=np.int_)
            prims[sl] = vertex_coords
            # Compute the normals to the triangles
            # Coordinate of the first vertices of every triangle making the
            # shape
            # [n_prims, xyz : 3]
            v0 = s.vertex_position(face_indices3.x)
            # Coordinate of the second vertices of every triangle making the
            # shape
            # [n_prims, xyz : 3]
            v1 = s.vertex_position(face_indices3.y)
            # Coordinate of the third vertices of every triangle making the
            # shape
            # [n_prims, xyz : 3]
            v2 = s.vertex_position(face_indices3.z)
            # Compute the normals
            # [n_prims, xyz : 3]
            mi_n = dr.normalize(dr.cross(v1 - v0, v2 - v0))
            # Move to TensorFlow
            # [n_prims, 3]
            n = mi_to_np_ndarray(mi_n, self._rdtype)
            # Update the 'normals' tensor
            normals[sl] = n

        self._primitives = prims
        self._normals = normals
        primitives_2_objects = np.asarray(primitives_2_objects, np.int_)
        self._primitives_2_objects = primitives_2_objects

        ####################################################
        # Used by the shoot & bounce method to map from
        # (shape, local primitive index) to the
        # corresponding global primitive index.
        ####################################################

        # [num_objects]
        self._prim_offsets = mi.Int32(prim_offsets)
        dest = dr.reinterpret_array_v(mi.UInt32, mi_scene.shapes_dr())
        if dr.width(dest) == 0:
            self._shape_indices = mi.Int32([])
        else:
            # [num_objects]
            shape_indices = dr.full(mi.Int32, -1, dr.max(dest)[0] + 1)
            dr.scatter(shape_indices, dr.arange(mi.Int32, 0, dr.width(dest)), dest)
            dr.eval(shape_indices)
            # [num_objects]
            self._shape_indices = shape_indices

        #################################################
        # Extract the wedges
        #################################################
        # _wedges_origin : [num_wedges, 3], float
        #   Starting point of the wedges

        # _wedges_e_hat : [num_wedges, 3], float
        #   Normalized edge vector

        # _wedges_length : [num_wedges], float
        #   Length of the wedges

        # _wedges_normals : [num_wedges, 2, 3], float
        #   Normals to the wedges sides

        # _primitives_2_wedges : [num_primitives, 3], int
        #   Maps primitives to their wedges

        # _wedges_objects : [num_wedges, 2], int
        #   Indices of the two objects making the wedge (the two sides of the
        #   wedge could belong to different objects)

        # is_edge : [num_wedges], bool
        #     Set to `True` if a wedge is an edge, i.e., the edge of a single
        #     primitive.

        edges = self._extract_wedges()
        self._wedges_origin = edges[0]
        self._wedges_e_hat = edges[1]
        self._wedges_length = edges[2]
        self._wedges_normals = edges[3]
        self._primitives_2_wedges = edges[4]
        self._wedges_objects = edges[5]
        self._is_edge = edges[6]

    ##################################################################
    # Internal utility methods
    ##################################################################

    @property
    def primitives(self):
        """
        [num triangles, 3, 3], np.float_ : The triangles: [x,y,z] coordinates of
        the 3 vertices of every triangle
        """
        return self._primitives

    @property
    def normals(self):
        """
        [num triangles, 3], np.float_ : The normals of the triangles
        """
        return self._normals

    @property
    def prim_offsets(self):
        """
        [num_objects], np.int_ : Indices offsets for accessing the triangles
        each shape in `primitives`
        """
        return self._prim_offsets

    @property
    def shape_indices(self):
        """
        [num_objects], np.int_ :  Map object ids to indices that can be used to
        access `prim_offsets`
        """
        return self._shape_indices

    @property
    def wedges_origin(self):
        """
        [num_wedges, 3], np.float_ : Origin of the wedges
        """
        return self._wedges_origin

    @property
    def wedges_e_hat(self):
        """
        [num_wedges, 3], np.float_ : Normalized edge vector
        """
        return self._wedges_e_hat

    @property
    def wedges_length(self):
        """
        [num_wedges], np.float_ : Length of the wedges
        """
        return self._wedges_length

    @property
    def wedges_normals(self):
        """
        [num_wedges, 2, 3], np.float_ : Normals to the wedges sides
        """
        return self._wedges_normals

    @property
    def primitives_2_wedges(self):
        """
        [num_primitives, 3], np.int_ : Maps primitives to their wedges
        """
        return self._primitives_2_wedges

    @property
    def wedges_objects(self):
        """
        [num_wedges, 2], np.int_ : Indices of the two objects making the
        wedge (the two sides of the wedge could belong to different objects)
        """
        return self._wedges_objects

    @property
    def is_edge(self):
        """
        [num_wedges], np.bool_ : Set to `True` if a wedge is an edge, i.e.,
        the edge of a single primitive.
        """
        return self._is_edge

    def _build_scene_object_properties_tensors(self):
        r"""
        Build tensor containing the shape properties

        Input
        ------
        None

        Output
        -------
        relative_permittivity : [num_shape], np.complex_
            Tensor containing the complex relative permittivities of all shapes

        scattering_coefficient : [num_shape], np.float_
            Tensor containing the scattering coefficients of all shapes

        xpd_coefficient : [num_shape], np.float_
            Tensor containing the cross-polarization discrimination
            coefficients of all shapes

        alpha_r : [num_shape], np.float_
            Tensor containing the alpha_r scattering parameters of all shapes

        alpha_i : [num_shape], np.float_
            Tensor containing the alpha_i scattering parameters of all shapes

        lambda_ : [num_shape], np.float_
            Tensor containing the lambda_ scattering parameters of all shapes

        velocity : [num_shape], np.float_
            Tensor containing the velocity vectors of all shapes
        """

        objects_id = dr.reinterpret_array_v(mi.UInt32, self._mi_scene.shapes_dr())

        # Compute the size of the parameter tensors for all object_ids
        # We start indexing at 1 (not 0) and need to add the number of
        # RIS is the system
        array_size = np.max(objects_id) + 1 + len(self._scene.ris)

        # If a callable is set to obtain radio material properties, then there
        # is no need to build the tensors of material properties
        rm_callable_set = self._scene.radio_material_callable is not None

        # If a callable is set to obtain scattering patterns, then there
        # is no need to build the tensors for scattering properties
        sp_callable_set = self._scene.scattering_pattern_callable is not None

        if rm_callable_set:
            relative_permittivity = np.zeros([0], self._dtype)
            scattering_coefficient = np.zeros([0], self._rdtype)
            xpd_coefficient = np.zeros([0], self._rdtype)
        else:
            relative_permittivity = np.zeros([array_size], self._dtype)
            scattering_coefficient = np.zeros([array_size], self._rdtype)
            xpd_coefficient = np.zeros([array_size], self._rdtype)

        if sp_callable_set:
            alpha_r = np.zeros([0], np.int_)
            alpha_i = np.zeros([0], np.int_)
            lambda_ = np.zeros([0], self._rdtype)
        else:
            alpha_r = np.zeros([array_size], np.int_)
            alpha_i = np.zeros([array_size], np.int_)
            lambda_ = np.zeros([array_size], self._rdtype)

        if (not sp_callable_set) or (not rm_callable_set):

            for rm in self._scene.radio_materials.values():
                using_objects = rm.using_objects
                num_using_objects = np.shape(using_objects)[0]
                if num_using_objects == 0:
                    continue

                def f(x):
                    return np.full([num_using_objects], x)
                idx = np.reshape(using_objects, [-1, 1])
                if not rm_callable_set:
                    relative_permittivity[idx] = f(rm.complex_relative_permittivity)
                    scattering_coefficient[idx] = f(rm.scattering_coefficient)
                    xpd_coefficient[idx] = f(rm.xpd_coefficient)
                else:
                    alpha_r[idx] = f(rm.scattering_pattern.alpha_r)
                    alpha_i[idx] = f(rm.scattering_pattern.alpha_i)
                    lambda_[idx] = f(rm.scattering_pattern.lambda_)

        # Velocity of all objects
        # [array_size, 3]
        velocity = np.zeros([array_size, 3], self._rdtype)
        for obj in self._scene.objects.values():
            velocity[obj.object_id] = obj.velocity
        for obj in self._scene.ris.values():
            velocity[obj.object_id] = obj.velocity

        return (
            relative_permittivity,
            scattering_coefficient,
            xpd_coefficient,
            alpha_r,
            alpha_i,
            lambda_,
            velocity,
        )

    def _test_obstruction(self, o, d, maxt, additional_blockers=None):
        r"""
        Test obstruction of a batch of rays using Mitsuba.

        Input
        -----
        o: [batch_size, 3], np.float_
            Origin of the rays

        d: [batch_size, 3], np.float_
            Direction of the rays.
            Must be unit vectors.

        maxt: [batch_size], np.float_
            Length of the ray

        additional_blockers : mi.Scene | mi.Shape | None
            Optional Mitsuba scene or shape containing additional blockers.
            Defaults to `None`.

        Output
        -------
        val: [batch_size], np.bool_
            `True` if the ray is obstructed, i.e., hits a primitive.
            `False` otherwise.
        """
        # Translate the origin a bit along the ray direction to avoid
        # consecutive intersection with the same primitive
        o = o + SolverBase.EPSILON_OBSTRUCTION * d
        # [batch_size, 3]
        mi_o = self._mi_point_t(o)
        # Ray direction
        # [batch_size, 3]
        mi_d = self._mi_vec_t(d)
        # [batch_size]
        # Reduce the ray length by a small value to avoid false positive when
        # testing for LoS to a primitive due to hitting the primitive we are
        # testing visibility to.
        maxt = maxt - 2.0 * SolverBase.EPSILON_OBSTRUCTION
        mi_maxt = self._mi_scalar_t(maxt)
        # Mitsuba ray
        mi_ray = mi.Ray3f(
            o=mi_o, d=mi_d, maxt=mi_maxt, time=0.0, wavelengths=mi.Color0f(0.0)
        )
        # Test for obstruction using Mitsuba
        # With the scene
        # [batch_size]
        mi_val = self._mi_scene.ray_test(mi_ray)
        val = mi_to_np_ndarray(mi_val, np.bool_)
        # With additional blockers
        if additional_blockers:
            mi_val = additional_blockers.ray_test(mi_ray)
            val = np.logical_or(val, mi_to_np_ndarray(mi_val, np.bool_))
        return val

    def _extract_wedges(self):
        r"""
        Extract the wedges and, optionally, the edges, from the scene geometry

        Output
        ------
        # _wedges_origin : [num_wedges, 3], float
        #   Starting point of the wedges

        # _wedges_e_hat : [num_wedges, 3], float
        #   Normalized edge vector

        # _wedges_length : [num_wedges], float
        #   Length of the wedges

        # _wedges_normals : [num_wedges, 2, 3], float
        #   Normals to the wedges sides

        # _primitives_2_wedges : [num_primitives, 3], int
        #   Maps primitives to their wedges

        # _wedges_objects : [num_wedges, 2], int
        #   Indices of the two objects making the wedge (the two sides of the
        #   wedge could belong to different objects)

        # is_edge : [num_wedges], bool
        #     Set to `True` if a wedge is an edge, i.e., the edge of a single
        #     primitive.
        """

        angle_threshold = SolverBase.WEDGES_ANGLE_THRESHOLD

        # Extract vertices of every triangle
        # [num_prim, 3]
        v0 = self._primitives[:, 0, :]
        v1 = self._primitives[:, 1, :]
        v2 = self._primitives[:, 2, :]

        # List all edges
        # [num_prim, 2 + 2 + 2, 3]
        all_edges_undirected = np.concatenate([v0, v1, v1, v2, v2, v0], axis=1)
        # [num_edges = num_prim*3, 2, 3]
        all_edges_undirected = np.reshape(
            all_edges_undirected, (3 * v0.shape[0], 2, 3)
        )
        # Edges are oriented such that identical edges have same orientation
        # [num_edges, 2, 3]
        all_edges = self._swap_edges(all_edges_undirected)

        # Remaining point in the triangle for each edge.
        # This will be used to compute the normals later
        # [num_prim, 3, 3]
        remaining_vertex = np.concatenate([v2, v0, v1], axis=1)
        # [num_edges, 3]
        remaining_vertex = np.reshape(remaining_vertex, [-1, 3])

        # Get unique edges, i.e., wihout duplicates
        # unique_edges : [num_unique_edges, 2, 3]
        # indices_of_unique : [num_edges], index of the edge in ``unique_edges``
        unique_edges, indices_of_unique = np.unique(all_edges, axis=[0], return_inverse=True)

        # Number of occurences of every unique edge
        # [num_unique_edges]
        _, unique_indices_count = np.unique(indices_of_unique, return_counts=True)

        # Flag indicating which edges shared by exactly one or two primitives,
        # i.e., edges or wedges
        # [num_unique_edges]
        is_selected = np.logical_or(
            np.equal(unique_indices_count, 1), np.equal(unique_indices_count, 2)
        )

        # The following tensor lists the index of the first
        # edge in ``all_edges`` that makes the wedge.
        # Note: In the presence of duplicate values in `indices_of_unique`
        # (i.e. our case), it is not deterministic which of them
        # `tensor_scatter_nd_update` will store into the result. That's okay.
        # [num_edges]
        seq = np.arange(indices_of_unique.shape[0])
        # [num_unique_edges]
        default = np.full((unique_edges.shape[0],), fill_value=-1)
        # [num_unique_edges]
        default[indices_of_unique] = seq
        all_edges_index_1 = default.copy()
        # Next, list the second primitive that the wedge is connected to.
        # -1 is used for edges defined by a single primitive (screens)
        # [num_edges]
        missing = np.full((indices_of_unique.shape[0],), fill_value=True)
        missing[all_edges_index_1] = False
        # [num_unique_edges]
        default[indices_of_unique[missing]] = seq[missing]
        all_edges_index_2 = default.copy()
        # Flag set to True if an edge is not a wedge, i.e., is attached to only
        # one primitive. This is the case for unique edges for which
        # ``all_edges_index_2`` is not set to -1
        # [num_unique_edges]
        is_edge = np.equal(all_edges_index_2, -1)
        # For edges, the unique edge primitive is set to the same value for both
        # ``all_edges_index_1`` and ``all_edges_index_2``. This will lead to
        # mapping to the same primitive
        # [num_unique_edges]
        all_edges_index_2 = np.where(is_edge, all_edges_index_1, all_edges_index_2)

        # Normals to the faces
        # We compute the normals such that they point in the same direction
        # of the space for both primitives making a wedge.
        # To that aim, the normal for the face 0(n) is defined as the
        # cross-product between:
        # - The edge vector
        # - The vector connecting a vertex of the edge to the third
        #   point of the triangle corresponding to the 0(n) face to whom this
        #   edge belongs.
        # Edge vertices
        # [num_unique_edges, 2, 3]
        vs = all_edges[all_edges_index_1]
        # [num_unique_edges, 3]
        v1 = vs[:, 0]
        v2 = vs[:, 1]
        # Edge vector
        # [num_unique_edges, 3]
        e = v2 - v1
        # Vertex on the 0 and n faces
        # [num_unique_edges, 3]
        vf1 = remaining_vertex[all_edges_index_1]
        vf2 = remaining_vertex[all_edges_index_2]
        # [num_unique_edges, 3]
        u_1, _ = normalize(vf1 - v1)
        u_2, _ = normalize(vf2 - v1)
        # [num_unique_edges, 3]
        n1, _ = normalize(cross(e, u_1))
        n2, _ = normalize(cross(u_2, e))

        # We flip the normals if necessary to ensure that they point towards
        # the "exterior" of the wedge, i.e., that the exterior angle is at least
        # pi.
        # To ensure that, we orient the normals such that u2 does not point
        # towards the half-space defined by n1.
        # [num_unique_edges]
        cos_angle = dot(u_2, n1)
        # Three cases:
        # * cos_angle > 0: u2 points towards the same half space as n1.
        #   We must flip n1 and n2.
        # * cos_angle < 0: u2 points towards the same half space as n1. Do nothing
        # * cos_angle = 0: u2 is orthogonal to the n1, i.e., the two primitives are
        # parallel. In that cases, either the wedge is an edge, or it is a flat
        # surface that must be discarded
        # [num_unique_edges]
        flip = np.where(
            np.greater(cos_angle, np.zeros_like(cos_angle)),
            -np.ones_like(cos_angle),
            np.ones_like(cos_angle),
        )
        # [num_unique_edges, 1]
        flip = np.expand_dims(flip, axis=1)
        # [num_unique_edges, 3]
        n1 = n1 * flip
        n2 = n2 * flip
        # Discard the wedges considered as flat, i.e., with an opening angle
        # close to np.pi up to `angle_threshold`
        # We use this observation to discard close-to-flat wedges.
        # cos_angle = dot(u_2, n1) = cos(theta)
        # where theta= angle(u_2, n1).
        # Then, we want:
        # pi/2 - angle_threshold < theta < pi/2 + angle_threshold
        # => cos(pi/2 + angle_threshold) < cos(theta)
        #                                       < cos(pi/2 - angle_threshold)
        # => -sin(angle_threshold) < cos_angle < sin(angle_threshold)
        # ()
        theshold = np.abs(np.sin(np.asarray(angle_threshold, self._rdtype)))
        # [num_unique_edges]
        is_selected_ = np.greater(np.abs(cos_angle), theshold)
        # Don't discard edges
        # [num_unique_edges]
        is_selected_ = np.logical_or(is_edge, is_selected_)
        # [num_unique_edges]
        is_selected = np.logical_and(is_selected, is_selected_)

        # Extract only the selected lanes
        # [num_selected_edges]
        selected_indices = np.argwhere(is_selected)
        # [num_selected_edges, 2, 3]
        selected_edges = unique_edges[is_selected]
        # [num_selected_edges, 3]
        selected_wedges_start = selected_edges[:, 0]
        selected_wedges_end = selected_edges[:, 1]
        # [num_selected_edges, 3]
        n1 = n1[is_selected]
        n2 = n2[is_selected]
        # [num_selected_edges, 2, 3]
        # n1: 0-face
        # n2: n-face
        normals = np.stack([n1, n2], axis=1)

        # Pre-compute a mapping from primitive index to (up to) three wedges.
        # Recall that by construction, `all_edges` is ordered by
        # primitive (3 rows per primitive).
        # Then, `indices_of_unique` gives the mapping from the row indices
        # of `all_edges` to the row indices of `unique_edges`.
        # Finally, a subset of `unique_edges` via the `is_double` mask.
        #
        # In the end, all we need to do is renumber the values of
        # `indices_of_unique` to refer to rows of `selected_edges` instead of
        # rows of `unique_edges`.
        if selected_indices.size != 0:
            seq = np.asarray(np.arange(selected_edges.shape[0]), dtype=np.int_)
            default[selected_indices] = seq
        unique_edge_index_to_double_edge_index = default
        # [num_prim, 3]
        prim_to_wedges = np.reshape(
            np.take(unique_edge_index_to_double_edge_index, indices_of_unique), (-1, 3)
        )

        # Indices of the objects to which each edge belongs
        # First, indices (in all_edges) of edges
        # [num_unique_edges, 2]
        wedges_indices = np.stack([all_edges_index_1, all_edges_index_2], axis=1)
        # Keep only the selected wedges
        # [num_selected_edges, 2]
        wedges_indices = wedges_indices[is_selected]
        # [num_selected_edges]
        is_edge = is_edge[is_selected]
        # Wedges index 2 primitive index
        # [num_selected_edges, 2]
        wedges_2_prim = wedges_indices // 3
        # Primitive index 2 object index
        # [num_selected_edges, 2]
        wedges_2_object = np.take(self._primitives_2_objects, wedges_2_prim)

        # Edges length and edge vector
        # The edge vector e_hat must be such that:
        #   normalize(n_0 x n_n) = e_hat,
        # where n_0 is the normal to the 0-face and n_n the normal to the
        # n-face
        # [num_selected_edges, 3]
        e_hat, _ = normalize(cross(normals[..., 0, :], normals[..., 1, :]))
        # Select the wedges' origin according to the normals
        # e_hat_ind: [num_selected_edges, 3]
        # length : [num_selected_edges]
        e_hat_ind, length = normalize(selected_wedges_end - selected_wedges_start)
        # [num_selected_edges]
        origin_indicator = dot(e_hat, e_hat_ind)
        # [num_selected_edges, 3]
        origin_indicator = np.expand_dims(origin_indicator, axis=1)
        origin = np.where(
            origin_indicator < 0, selected_wedges_end, selected_wedges_start
        )
        # Set arbitrarely the vector for the edges
        # [num_selected_edges, 3]
        e_hat = np.where(np.expand_dims(is_edge, axis=1), e_hat_ind, e_hat)

        # Output
        output = (
            origin,  # wedges origins
            e_hat,  # Wedge vector
            length,  # Wedge length
            normals,  # wedges_normals
            prim_to_wedges,  # primitives_2_wedges
            wedges_2_object,  # wedges_objects
            is_edge,  # is_edge
        )

        return output

    def _swap_edges(self, edges):
        """Swap edges extremities such that identical edges are oriented in
        the same way.

        Parameters
        ----------
        edges : [...,2,3], float
            Batch of edges extremities

        Returns
        -------
        [..., 2, 3], float
            Reoriented edges
        """
        p0 = edges[:, 0, :]
        p1 = edges[:, 1, :]
        p0_hat, r0 = normalize(p0)
        p1_hat, r1 = normalize(p1)
        theta0, phi0 = theta_phi_from_unit_vec(p0_hat)
        theta1, phi1 = theta_phi_from_unit_vec(p1_hat)

        # Three considtions are used to orientate the edges
        # by swapping the extremities (p0,p1).
        # Condition n+1 is used only if none of the previous n conditions enabled
        # to separate the edges.
        # 1. norm(p1) >= norm(p0)
        # 2. azimuth(p1) >= azimuth(p0)
        # 3. elevation (p1) >= elevation(p0)

        # More details of the algorithm:
        # needs_swap 1: !r_equal and r0 > r1
        # needs_swap 2: r_equal and !phi_equal and phi0 > phi1
        # needs_swap 3: r_equal and phi_equal and theta0 > theta1
        # Note: case when all three coordinates are equal is not considered

        r_equal = np.isclose(r0, r1)
        phi_equal = np.isclose(phi0, phi1)
        case_2 = np.logical_and(r_equal, np.logical_not(phi_equal))
        case_3 = np.logical_and(r_equal, phi_equal)

        needs_swap_1 = np.logical_and(np.logical_not(r_equal), r0 > r1)
        needs_swap_2 = np.logical_and(case_2, phi0 > phi1)
        needs_swap_3 = np.logical_and(case_3, theta0 > theta1)

        needs_swap = np.any(
            np.stack([needs_swap_1, needs_swap_2, needs_swap_3], axis=1),
            keepdims=True,
            axis=1,
        )

        result = np.concatenate(
            [
                np.expand_dims(np.where(needs_swap, p1, p0), axis=1),
                np.expand_dims(np.where(needs_swap, p0, p1), axis=1),
            ],
            axis=1,
        )
        return result

    def _wedges_from_primitives(self, candidates, edge_diffraction):
        r"""
        Returns the candidate wedges from the candidate primitives.

        As only first-order diffraction is considered, only the wedges of the
        primitives in line-of-sight of the transmitter are considered.

        Input
        ------
        candidates: [max_depth, num_samples], int
            Candidate paths with depth up to ``max_depth``.
            Entries correspond to primitives indices.
            For paths with depth lower than ``max_depth``, -1 is used as
            padding value.
            The first path is the LoS one.

        edge_diffraction : bool
            If set to `False`, only diffraction on wedges, i.e., edges that
            connect two primitives, is considered.

        Output
        -------
        candidate_wedges : [num_candidate_wedges], int
            Candidate wedges.
            Entries correspond to wedges indices.
        """

        # If no candidates, return an empty list
        # Useful to manage empty scenes
        if candidates.shape[0] == 0:
            return np.asarray([], np.int_)

        # Remove -1
        candidates = np.take(candidates, np.argwhere(np.not_equal(candidates, -1))[:, 0])

        # Remove duplicates
        candidates, _ = np.unique(candidates)

        # [num_samples, 3]
        candidate_wedges = np.take(self._primitives_2_wedges, candidates, axis=0)
        # [num_samples*3]
        candidate_wedges = np.reshape(candidate_wedges, [-1])

        # Remove -1
        # [<= num_samples*3]
        candidate_wedges = np.take(
            candidate_wedges, np.argwhere(np.not_equal(candidate_wedges, -1))[:, 0]
        )

        # Remove duplicates
        # [num_candidate_wedges]
        candidate_wedges, _ = np.unique(candidate_wedges)

        # Remove edges if required
        if not edge_diffraction:
            # [num_candidate_wedges]
            is_wedge = ~np.take(self._is_edge, candidate_wedges)
            wedge_indices = np.argwhere(is_wedge)[:, 0]
            # [num_candidate_wedges]
            candidate_wedges = np.take(candidate_wedges, wedge_indices)

        return candidate_wedges

    def _build_mi_ris_objects(self):
        r"""
        Builds a Mitsuba scene containing all RIS as rectangles with position,
        orientation, and size matching the RIS properties.∂

        Output
        ------
        : mi.Scene
            Mitsuba Scene containing rectangles corresponding to RIS
        """
        # List of all the RIS objects in the scene
        all_ris = list(self._scene.ris.values())

        # Creates a scene containing RIS as rectangles
        scene_dict = {"type": "scene"}
        for ris in all_ris:
            center = ris.position
            orientation = ris.orientation
            size = ris.size
            mi_to_world = mitsuba_rectangle_to_world(
                center, orientation, size, ris=True
            )
            scene_dict[ris.name] = {"type": "rectangle", "to_world": mi_to_world}
        mi_ris_scene = mi.load_dict(scene_dict)

        return mi_ris_scene
