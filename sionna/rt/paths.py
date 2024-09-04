#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Dataclass that stores paths
"""

import os
import numpy as np

from . import scene as scene_module
from .utils import dot, r_hat


class Paths:
    # pylint: disable=line-too-long
    r"""
    Paths()

    Stores the simulated propagation paths

    Paths are generated for the loaded scene using
    :meth:`~sionna.rt.Scene.compute_paths`. Please refer to the
    documentation of this function for further details.
    These paths can then be used to compute channel impulse responses:

    .. code-block:: Python

        paths = scene.compute_paths()
        a, tau = paths.cir()

    where ``scene`` is the :class:`~sionna.rt.Scene` loaded using
    :func:`~sionna.rt.load_scene`.
    """

    # Input
    # ------

    # mask : (num_rx, num_tx, max_num_paths), np.bool_
    #   Set to `False` for non-existent paths.
    #   When there are multiple transmitters or receivers, path counts may vary between links. This is used to identify non-existent paths.
    #   For such paths, the channel coefficient is set to `0` and the delay to `-1`.

    # a : (num_rx, num_tx, max_num_paths, num_time_steps), np.complex_
    #     Channel coefficients :math:`a_i` as defined in :eq:`T_tilde`.
    #     If there are less than `max_num_path` valid paths between a
    #     transmit and receive antenna, the irrelevant elements are
    #     filled with zeros.

    # tau : (num_rx, num_tx, max_num_paths), np.float_
    #     Propagation delay of each path [s].
    #     If :attr:`~Scene.synthetic_array` is `True`, the shape of this tensor
    #     is `(1, num_rx, num_tx, max_num_paths)` as the delays for the
    #     individual antenna elements are assumed to be equal.
    #     If there are less than `max_num_path` valid paths between a
    #     transmit and receive antenna, the irrelevant elements are
    #     filled with -1.

    # theta_t : (num_rx, num_tx, max_num_paths), np.float_
    #     Zenith  angles of departure :math:`\theta_{\text{T},i}` [rad].
    #     If :attr:`~Scene.synthetic_array` is `True`, the shape of this tensor
    #     is `(1, num_rx, num_tx, max_num_paths)` as the angles for the
    #     individual antenna elements are assumed to be equal.

    # phi_t : (num_rx, num_tx, max_num_paths), np.float_
    #     Azimuth angles of departure :math:`\varphi_{\text{T},i}` [rad].
    #     See description of ``theta_t``.

    # theta_r : (num_rx, num_tx, max_num_paths), np.float_
    #     Zenith angles of arrival :math:`\theta_{\text{R},i}` [rad].
    #     See description of ``theta_t``.

    # phi_r : (num_rx, num_tx, max_num_paths), np.float_
    #     Azimuth angles of arrival :math:`\varphi_{\text{T},i}` [rad].
    #     See description of ``theta_t``.

    # types : (max_num_paths,), np.int_
    #     Type of path:

    #     - 0 : LoS
    #     - 1 : Reflected
    #     - 2 : Diffracted
    #     - 3 : Scattered

    # Types of paths
    LOS = 0
    SPECULAR = 1
    DIFFRACTED = 2
    SCATTERED = 3
    RIS = 4

    def __init__(self, sources, targets, scene, types=None):

        num_sources = sources.shape[0]
        num_targets = targets.shape[0]

        self._a = np.zeros([num_targets, num_sources, 0], np.complex_)
        self._tau = np.zeros([num_targets, num_sources, 0], np.float_)
        self._theta_t = np.zeros([num_targets, num_sources, 0], np.float_)
        self._theta_r = np.zeros([num_targets, num_sources, 0], np.float_)
        self._phi_t = np.zeros([num_targets, num_sources, 0], np.float_)
        self._phi_r = np.zeros([num_targets, num_sources, 0], np.float_)
        self._mask = np.full([num_targets, num_sources, 0], False)
        self._targets_sources_mask = np.fill([num_targets, num_sources, 0], False)
        self._vertices = np.zeros([0, num_targets, num_sources, 0, 3], np.float_)
        self._objects = np.full([0, num_targets, num_sources, 0], -1)
        self._doppler = np.zeros([num_targets, num_sources, 0], np.float_)
        if types is None:
            self._types = np.full([0], -1)
        else:
            self._types = types

        self._sources = sources
        self._targets = targets
        self._scene = scene

        # Is the direction reversed?
        self._reverse_direction = False
        # Normalize paths delays?
        self._normalize_delays = False

    def to_dict(self):
        # pylint: disable=line-too-long
        r"""
        Returns the properties of the paths as a dictionary which values are
        tensors

        Output
        -------
        : `dict`
        """
        members_names = dir(self)
        members_objects = [getattr(self, attr) for attr in members_names]
        data = {
            attr_name[1:]: attr_obj
            for (attr_obj, attr_name) in zip(members_objects, members_names)
            if not callable(attr_obj)
            and not isinstance(attr_obj, scene_module.Scene)
            and not attr_name.startswith("__")
            and attr_name.startswith("_")
        }
        return data

    def from_dict(self, data_dict):
        # pylint: disable=line-too-long
        r"""
        Set the paths from a dictionary which values are tensors

        The format of the dictionary is expected to be the same as the one
        returned by :meth:`~sionna.rt.Paths.to_dict()`.

        Input
        ------
        data_dict : `dict`
        """
        for attr_name in data_dict:
            attr_obj = data_dict[attr_name]
            setattr(self, "_" + attr_name, attr_obj)

    def export(self, filename):
        r"""
        export(filename)

        Saves the paths as an OBJ file for visualisation, e.g., in Blender

        Input
        ------
        filename : str
            Path and name of the file
        """
        vertices = self.vertices
        objects = self.objects
        sources = self.sources
        targets = self.targets
        mask = self.targets_sources_mask

        # Content of the obj file
        r = ""
        offset = 0
        for rx in range(vertices.shape[1]):
            tgt = targets[rx].numpy()
            for tx in range(vertices.shape[2]):
                src = sources[tx].numpy()
                for p in range(vertices.shape[3]):

                    # If the path is masked, skip it
                    if not mask[rx, tx, p]:
                        continue

                    # Add a comment to describe this path
                    r += f"# Path {p} from tx {tx} to rx {rx}" + os.linesep
                    # Vertices and intersected objects
                    vs = vertices[:, rx, tx, p]
                    objs = objects[:, rx, tx, p]

                    depth = 0
                    # First vertex is the source
                    r += f"v {src[0]:.8f} {src[1]:.8f} {src[2]:.8f}" + os.linesep
                    # Add intersection points
                    for v, o in zip(vs, objs):
                        # Skip if no intersection
                        if o == -1:
                            continue
                        r += f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}" + os.linesep
                        depth += 1
                    r += f"v {tgt[0]:.8f} {tgt[1]:.8f} {tgt[2]:.8f}" + os.linesep

                    # Add the connections
                    for i in range(1, depth + 2):
                        v0 = i + offset
                        v1 = i + offset + 1
                        r += f"l {v0} {v1}" + os.linesep

                    # Prepare for the next path
                    r += os.linesep
                    offset += depth + 2

        # Save the file
        # pylint: disable=unspecified-encoding
        with open(filename, "w") as f:
            f.write(r)

    @property
    def mask(self):
        # pylint: disable=line-too-long
        """
        (num_rx, num_tx, max_num_paths), np.bool_ : Set to `False` for non-existent paths.
        When there are multiple transmitters or receivers, path counts may vary between links. This is used to identify non-existent paths.
        For such paths, the channel coefficient is set to `0` and the delay to `-1`.
        """
        return self._mask

    @mask.setter
    def mask(self, v):
        self._mask = v

    @property
    def a(self):
        # pylint: disable=line-too-long
        """
        (num_rx, num_tx, max_num_paths, num_time_steps), np.complex_ : Passband channel coefficients :math:`a_i` of each path as defined in :eq:`H_final`.
        """
        return self._a

    @a.setter
    def a(self, v):
        self._a = v

    @property
    def tau(self):
        # pylint: disable=line-too-long
        """
        (num_rx, num_tx, max_num_paths), np.float_ : Propagation delay :math:`\\tau_i` [s] of each path as defined in :eq:`H_final`.
        """
        return self._tau

    @tau.setter
    def tau(self, v):
        self._tau = v

    @property
    def theta_t(self):
        # pylint: disable=line-too-long
        """
        (num_rx, num_tx, max_num_paths), np.float_ : Zenith  angles of departure [rad]
        """
        return self._theta_t

    @theta_t.setter
    def theta_t(self, v):
        self._theta_t = v

    @property
    def phi_t(self):
        # pylint: disable=line-too-long
        """
        (num_rx, num_tx, max_num_paths), np.float_ : Azimuth angles of departure [rad]
        """
        return self._phi_t

    @phi_t.setter
    def phi_t(self, v):
        self._phi_t = v

    @property
    def theta_r(self):
        # pylint: disable=line-too-long
        """
        (num_rx, num_tx, max_num_paths), np.float_ : Zenith angles of arrival [rad]
        """
        return self._theta_r

    @theta_r.setter
    def theta_r(self, v):
        self._theta_r = v

    @property
    def phi_r(self):
        # pylint: disable=line-too-long
        """
        (num_rx, num_tx, max_num_paths), np.float_ : Azimuth angles of arrival [rad]
        """
        return self._phi_r

    @phi_r.setter
    def phi_r(self, v):
        self._phi_r = v

    @property
    def types(self):
        """
        (max_num_paths,), np.int_ : Type of the paths:

        - 0 : LoS
        - 1 : Reflected
        - 2 : Diffracted
        - 3 : Scattered
        - 4 : RIS
        """
        return self._types

    @types.setter
    def types(self, v):
        self._types = v

    @property
    def sources(self):
        # pylint: disable=line-too-long
        """
        (num_sources, 3), np.float_ : Sources from which rays (paths) are emitted
        """
        return self._sources

    @sources.setter
    def sources(self, v):
        self._sources = v

    @property
    def targets(self):
        # pylint: disable=line-too-long
        """
        (num_targets, 3), np.float_ : Targets at which rays (paths) are received
        """
        return self._targets

    @targets.setter
    def targets(self, v):
        self._targets = v

    @property
    def normalize_delays(self):
        """
        bool : Set to `True` to normalize path delays such that the first path
        between any pair of antennas of a transmitter and receiver arrives at
        ``tau = 0``. Defaults to `True`.
        """
        return self._normalize_delays

    @normalize_delays.setter
    def normalize_delays(self, v):
        if v == self._normalize_delays:
            return

        if ~v and self._normalize_delays:
            self.tau += self._min_tau
        else:
            self.tau -= self._min_tau
        self.tau = np.where(self.tau < 0, np.astype(-1, self.tau.dtype), self.tau)
        self._normalize_delays = v

    @property
    def doppler(self):
        # pylint: disable=line-too-long
        """
        (num_rx, num_tx, max_num_paths), np.float_ : Doppler shift for each path
        related to movement of objects. The Doppler shifts resulting from
        movements of the transmitters or receivers will be computed from the inputs to the function :func:`~sionna.rt.Paths.apply_doppler`.
        """
        return self._doppler

    @doppler.setter
    def doppler(self, v):
        self._doppler = v

    def apply_doppler(
        self,
        sampling_frequency,
        num_time_steps,
        tx_velocities=(0.0, 0.0, 0.0),
        rx_velocities=(0.0, 0.0, 0.0),
    ):
        # pylint: disable=line-too-long
        r"""
        Apply Doppler shifts to all paths according to the velocities
        of objects in the scene as well as the provided transmitter and receiver velocities.

        This function replaces the last dimension of the tensor :attr:`~sionna.rt.Paths.a` storing the
        time evolution of the paths' coefficients with a dimension of size ``num_time_steps``.

        Time evolution of the channel coefficients is simulated by computing the
        Doppler shift due to movements of scene objects, transmitters, and receivers.
        To understand this process, let us consider a single propagation path undergoing
        :math:`n` scattering processes, such as reflection, diffuse scattering, or diffraction,
        as shown in the figure below.

        .. figure:: ../figures/doppler.png
            :align: center

        The object on which lies the :math:`i\text{th}` scattering point has the velocity vector
        :math:`\hat{\mathbf{v}}_i` and the outgoing ray direction at this point is
        denoted :math:`\hat{\mathbf{k}}_i`. The first and last point correspond to the transmitter
        and receiver, respectively. We therefore have

        .. math::

            \hat{\mathbf{k}}_0 &= \hat{\mathbf{r}}(\theta_{\text{T}}, \varphi_{\text{T}})\\
            \hat{\mathbf{k}}_{n} &= -\hat{\mathbf{r}}(\theta_{\text{R}}, \varphi_{\text{R}})

        where :math:`(\theta_{\text{T}}, \varphi_{\text{T}})` are the AoDs,
        :math:`(\theta_{\text{R}}, \varphi_{\text{R}})` are the AoAs, and :math:`\hat{\mathbf{r}}(\theta,\varphi)` is defined in :eq:`spherical_vecs`.

        If the transmitter emits a signal with frequency :math:`f`, the receiver
        will observe the signal at frequency :math:`f'=f + f_\Delta`, where :math:`f_\Delta` is the Doppler
        shift, which can be computed as [Wiffen2018]_

        .. math::

            f' = f \prod_{i=0}^n \frac{1 - \frac{\mathbf{v}_{i+1}^\mathsf{T}\hat{\mathbf{k}}_i}{c}}{1 - \frac{\mathbf{v}_{i}^\mathsf{T}\hat{\mathbf{k}}_i}{c}}.

        Under the assumption that :math:`\lVert \mathbf{v}_i \rVert\ll c`, we can apply the Taylor expansion :math:`(1-x)^{-1}\approx 1+x`, for :math:`x\ll 1`, to the previous equation
        to obtain

        .. math::

            f' &\approx f \prod_{i=0}^n \left(1 - \frac{\mathbf{v}_{i+1}^\mathsf{T}\hat{\mathbf{k}}_i}{c}\right)\left(1 + \frac{\mathbf{v}_{i}^\mathsf{T}\hat{\mathbf{k}}_i}{c}\right)\\
               &\approx f \left(1 + \sum_{i=0}^n \frac{\mathbf{v}_{i}^\mathsf{T}\hat{\mathbf{k}}_i -\mathbf{v}_{i+1}^\mathsf{T}\hat{\mathbf{k}}_i}{c} \right)

        where the second line results from ignoring terms in :math:`c^{-2}`. Solving for :math:`f_\Delta`, grouping terms with the same :math:`\mathbf{v}_i` together, and using :math:`f=c/\lambda`, we obtain

        .. math::

            f_\Delta = \frac{1}{\lambda}\left(\mathbf{v}_{0}^\mathsf{T}\hat{\mathbf{k}}_0 - \mathbf{v}_{n+1}^\mathsf{T}\hat{\mathbf{k}}_n + \sum_{i=1}^n \mathbf{v}_{i}^\mathsf{T}\left(\hat{\mathbf{k}}_i-\hat{\mathbf{k}}_{i-1} \right) \right) \qquad \text{[Hz]}.

        Using this Doppler shift, the time-dependent path coefficient is computed as

        .. math ::

            a(t) = a e^{j2\pi f_\Delta t}.

        Note that this model is only valid as long as the AoDs, AoAs, and path delays do not change significantly.
        This is typically the case for very short time intervals. Large-scale mobility should be simulated by moving objects within the scene and recomputing the propagation paths.

        When this function is called multiple times, it overwrites the previous time step dimension.

        Input
        ------
        sampling_frequency : float
            Frequency [Hz] at which the channel impulse response is sampled

        num_time_steps : int
            Number of time steps.

        tx_velocities : (num_tx, 3), np.float_
            Velocity vectors :math:`(v_\text{x}, v_\text{y}, v_\text{z})` of all
            transmitters [m/s].
            Defaults to `[0,0,0]`.

        rx_velocities : (num_tx, 3), np.float_
            Velocity vectors :math:`(v_\text{x}, v_\text{y}, v_\text{z})` of all
            receivers [m/s].
            Defaults to `[0,0,0]`.
        """

        two_pi = 2.0 * np.pi

        if tx_velocities.shape[1] != 3:
            raise ValueError("Last dimension of `tx_velocities` must equal 3")
        if rx_velocities.shape[1] != 3:
            raise ValueError("Last dimension of `rx_velocities` must equal 3")
        if sampling_frequency <= 0.0:
            raise ValueError("The sampling frequency must be positive")
        if num_time_steps <= 0:
            raise ValueError("The number of time samples must a positive integer")

        # Drop previous time step dimension, if any
        if self.a.ndim == 4:
            self.a = self.a[:, :, :, 0]

        # (num_rx, num_tx, max_num_paths, 3)
        k_t = r_hat(self.theta_t, self.phi_t)
        k_r = r_hat(self.theta_r, self.phi_r)

        # Expand rank of the speed vector for broadcasting with k_r
        # (1, num_tx, 1, 3)
        tx_velocities = np.expand_dims(tx_velocities, (0, 1, 3, 4))
        # (num_rx,  1, 1, 3)
        rx_velocities = np.expand_dims(rx_velocities, (1, 2, 3, 4))

        # Generate time steps
        # (num_time_steps,)
        ts = np.arange(num_time_steps, dtype=np.float_)
        ts = ts / sampling_frequency

        # Compute the Doppler shift
        # (num_rx, num_tx, max_num_paths)
        tx_ds = two_pi * dot(tx_velocities, k_t) / self._scene.wavelength
        rx_ds = two_pi * dot(rx_velocities, k_r) / self._scene.wavelength
        ds = tx_ds + rx_ds

        # Add Doppler shifts due to movement of scene objects
        ds += two_pi * self.doppler

        # Expand for the time sample dimension
        # (num_rx, num_tx, max_num_paths, 1)
        ds = np.expand_dims(ds, axis=-1)
        # Expand time steps for broadcasting
        # (1, 1, 1, num_time_steps)
        ts = ts.reshape((1, 1, 1, ts.size))
        # (num_rx, num_tx, max_num_paths, num_time_steps)
        ds = ds * ts
        exp_ds = np.exp(1j * ds)

        # Apply Doppler shift
        # Expand with time dimension
        # (num_rx, num_tx, max_num_paths, 1)
        a = np.expand_dims(self.a, axis=-1)

        # Manual broadcast last dimension
        a = np.repeat(a, exp_ds.shape[-1], -1)

        a = a * exp_ds

        self.a = a

    def cir(
        self,
        los=True,
        reflection=True,
        diffraction=True,
        scattering=True,
        ris=True,
        cluster_ris_paths=True,
        num_paths=None,
    ):
        # pylint: disable=line-too-long
        r"""
        Returns the baseband equivalent channel impulse response :eq:`h_b`
        which can be used for link simulations by other Sionna components.

        The baseband equivalent channel coefficients :math:`a^{\text{b}}_{i}`
        are computed as :

        .. math::
            a^{\text{b}}_{i} = a_{i} e^{-j2 \pi f \tau_{i}}

        where :math:`i` is the index of an arbitrary path, :math:`a_{i}`
        is the passband path coefficient (:attr:`~sionna.rt.Paths.a`),
        :math:`\tau_{i}` is the path delay (:attr:`~sionna.rt.Paths.tau`),
        and :math:`f` is the carrier frequency.

        Note: For the paths of a given type to be returned (LoS, reflection, etc.), they
        must have been previously computed by :meth:`~sionna.rt.Scene.compute_paths`, i.e.,
        the corresponding flags must have been set to `True`.

        Input
        ------
        los : bool
            If set to `False`, LoS paths are not returned.
            Defaults to `True`.

        reflection : bool
            If set to `False`, specular paths are not returned.
            Defaults to `True`.

        diffraction : bool
            If set to `False`, diffracted paths are not returned.
            Defaults to `True`.

        scattering : bool
            If set to `False`, scattered paths are not returned.
            Defaults to `True`.

        ris : bool
            If set to `False`, paths involving RIS are not returned.
            Defaults to `True`.

        cluster_ris_paths : bool
            If set to `True`, the paths from each RIS are coherently combined
            into a single path, and the delays are averaged.
            Note that this process is performed separately for each RIS.
            For large RIS, clustering the paths significantly reduces the memory
            required to run link-level simulations.
            Defaults to `True`.

        num_paths : int or `None`
            All CIRs are either zero-padded or cropped to the largest
            ``num_paths`` paths.
            Defaults to `None` which means that no padding or cropping is done.

        Output
        -------
        a : (num_rx, num_tx, max_num_paths, num_time_steps), np.complex_
            Path coefficients

        tau : (num_rx, num_tx, max_num_paths), np.float_
            Path delays
        """

        ris = ris and (len(self._scene.ris) > 0)

        # Select only the desired effects
        types = self.types
        # (max_num_paths,)
        selection_mask = np.full(types.shape, False)
        if los:
            selection_mask = np.logical_or(selection_mask | types == Paths.LOS)
        if reflection:
            selection_mask = np.logical_or(selection_mask, types == Paths.SPECULAR)
        if diffraction:
            selection_mask = np.logical_or(selection_mask, types == Paths.DIFFRACTED)
        if scattering:
            selection_mask = np.logical_or(selection_mask, types == Paths.SCATTERED)
        if ris:
            if cluster_ris_paths:
                # Combine path coefficients from every RIS coherently and
                # average their delays.
                # This process is performed separately for each RIS.
                #
                # Extract paths coefficients and delays corresponding to RIS
                # (num_rx, num_tx, num_ris_paths, num_time_steps)
                a_ris = self.a[:, :, types == Paths.RIS, :]
                # (num_rx, num_tx, num_ris_paths)
                tau_ris = self.tau[:, :, types == Paths.RIS]
                # Loop over RIS to combine their path coefficients and delays
                index_start = 0
                index_end = 0
                a_combined_ris_all = []
                tau_combined_ris_all = []
                for ris_ in self._scene.ris.values():
                    index_end = index_start + ris_.num_cells
                    # Extract the path coefficients and delays corresponding to
                    # the paths from RIS
                    # (num_rx, num_tx, num_this_ris_path, num_time_steps)
                    a_this_ris = a_ris[..., index_start:index_end, :]
                    # (num_rx, num_tx, num_this_ris_path)
                    tau_this_ris = tau_ris[..., index_start:index_end]
                    # Average the delays
                    # (num_rx, num_tx, 1)
                    mean_tau_this_ris = np.mean(tau_this_ris, axis=-1, keepdims=True)
                    # Phase shift due to propagation delay.
                    # We subtract the average delay to ensure the propagation
                    # delay is not applied, only the phase shift due to the
                    # RIS geometry
                    # (num_rx, num_tx, num_this_ris_path)
                    tau_this_ris -= mean_tau_this_ris
                    ps = (
                        np.zeros_like(tau_this_ris)
                        - 2.0j * np.pi * self._scene.frequency * tau_this_ris
                    )
                    ps = ps[..., np.newaxis]
                    # (num_rx, num_tx, num_this_ris_path, num_time_steps)
                    a_this_ris = a_this_ris * np.exp(ps)
                    # Combine the paths coefficients and delays
                    # (num_rx, num_tx, 1, num_time_steps)
                    a_this_ris = np.sum(a_this_ris, axis=-2, keepdims=True)

                    #
                    a_combined_ris_all.append(a_this_ris)
                    tau_combined_ris_all.append(mean_tau_this_ris)
                    #
                    index_start = index_end
                #
                # (num_rx, num_tx, num_ris, num_time_steps)
                a_combined_ris_all = np.concatenate(a_combined_ris_all, axis=-2)
                # (num_rx, num_tx, num_ris)
                tau_combined_ris_all = np.concatenate(tau_combined_ris_all, axis=-1)
            else:
                selection_mask = np.logical_or(selection_mask, types == Paths.RIS)

        # Extract selected paths
        # (num_rx, num_tx, num_selected_paths, num_time_steps)
        a = self.a[:, :, selection_mask, :]
        # (num_rx, num_tx, num_selected_paths)
        tau = self.tau[:, :, selection_mask]

        # If RIS paths were combined, add the results of the clustering
        if ris and cluster_ris_paths:
            # (num_rx, num_tx, num_selected_paths, num_time_steps)
            a = np.concatenate([a, a_combined_ris_all], axis=-2)
            # (num_rx, num_tx, num_selected_paths)
            tau = np.concatenate([tau, tau_combined_ris_all], axis=-1)

        # Compute base-band CIR
        # (num_rx, num_tx, num_selected_paths, 1)
        tau = np.expand_dims(tau, -1)
        phase = np.zeros_like(tau) - 2.0j * np.pi * self._scene.frequency * tau
        # Manual repeat along the time step dimension as high-dimensional
        # broadcast is not possible
        phase = np.repeat(phase, a.shape[-1], axis=-1)
        a = a * np.exp(phase)

        if num_paths is not None:
            a, tau = self.pad_or_crop(a, tau, num_paths)

        return a, tau

    #######################################################
    # Internal methods and properties
    #######################################################

    @property
    def targets_sources_mask(self):
        # pylint: disable=line-too-long
        """
        (num_targets, num_sources, max_num_paths), np.bool_ : Set to `False` for non-existent paths.
        When there are multiple transmitters or receivers, path counts may vary between links. This is used to identify non-existent paths.
        For such paths, the channel coefficient is set to `0` and the delay to `-1`.
        Same as `mask`, but for sources and targets.
        """
        return self._targets_sources_mask

    @targets_sources_mask.setter
    def targets_sources_mask(self, v):
        self._targets_sources_mask = v

    @property
    def vertices(self):
        # pylint: disable=line-too-long
        """
        (max_depth, num_targets, num_sources, max_num_paths, 3), np.float_ : Positions of intersection points.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = v

    @property
    def objects(self):
        # pylint: disable=line-too-long
        """
        (max_depth, num_targets, num_sources, max_num_paths), np.int_ : Indices of the intersected scene objects
        or wedges. Paths with depth lower than ``max_depth`` are padded with `-1`.
        """
        return self._objects

    @objects.setter
    def objects(self, v):
        self._objects = v

    def merge(self, more_paths):
        r"""
        Merge ``more_paths`` with the current paths and returns the so-obtained
        instance. `self` is not updated.

        Input
        -----
        more_paths : :class:`~sionna.rt.Paths`
            First set of paths to merge
        """

        more_vertices = more_paths.vertices
        more_objects = more_paths.objects
        more_types = more_paths.types

        # The paths to merge must have the same number of sources and targets
        if more_paths.sources.shape[0] != self.sources.shape[0]:
            raise ValueError("Paths to merge must have same number of sources")
        if more_paths.sources.shape[0] != self.sources.shape[0]:
            raise ValueError("Paths to merge must have same number of targets")

        # Pad the paths with the lowest depth
        padding = self.vertices.shape[0] - more_vertices.shape[0]
        if padding > 0:
            more_vertices = np.pad(
                more_vertices,
                [[0, padding], [0, 0], [0, 0], [0, 0], [0, 0]],
                constant_values=0.0,
            )
            more_objects = np.pad(
                more_objects, [[0, padding], [0, 0], [0, 0], [0, 0]], constant_values=-1
            )
        elif padding < 0:
            padding = -padding
            self.vertices = np.pad(
                self.vertices,
                [[0, padding], [0, 0], [0, 0], [0, 0], [0, 0]],
                constant_values=0.,
            )
            self.objects = np.pad(
                self.objects, [[0, padding], [0, 0], [0, 0], [0, 0]], constant_values=-1
            )

        # Merge types
        if self.types.ndim == 0:
            merged_types = np.repeat(self.types, self.vertices.shape[3])
        else:
            merged_types = self.types
        if more_types.ndim == 0:
            more_types = np.repeat(more_types, more_vertices.shape[3])

        self.types = np.concatenate([merged_types, more_types], axis=0)

        # Concatenate all
        self.a = np.concatenate([self.a, more_paths.a], axis=2)
        self.tau = np.concatenate([self.tau, more_paths.tau], axis=2)
        self.theta_t = np.concatenate([self.theta_t, more_paths.theta_t], axis=2)
        self.phi_t = np.concatenate([self.phi_t, more_paths.phi_t], axis=2)
        self.theta_r = np.concatenate([self.theta_r, more_paths.theta_r], axis=2)
        self.phi_r = np.concatenate([self.phi_r, more_paths.phi_r], axis=2)
        self.mask = np.concatenate([self.mask, more_paths.mask], axis=2)
        self.vertices = np.concatenate([self.vertices, more_vertices], axis=3)
        self.objects = np.concatenate([self.objects, more_objects], axis=3)
        self.doppler = np.concatenate([self.doppler, more_paths.doppler], axis=2)

        return self

    def finalize(self):
        """
        This function must be called to finalize the creation of the paths.
        This function:

        - Flags the LoS paths

        - Computes the smallest delay for delay normalization
        """

        self.set_los_path_type()

        # Add dummy-dimension for batch_size
        # (1, num_rx, num_tx, max_num_paths)
        self.mask = np.expand_dims(self.mask, axis=0)
        # (1, num_rx, num_tx, max_num_paths)
        self.a = np.expand_dims(self.a, axis=0)
        # (1, num_rx, num_tx, max_num_paths)
        self.tau = np.expand_dims(self.tau, axis=0)
        # (1, num_rx, num_tx, max_num_paths)
        self.theta_t = np.expand_dims(self.theta_t, axis=0)
        # (1, num_rx, num_tx, max_num_paths)
        self.phi_t = np.expand_dims(self.phi_t, axis=0)
        # (1, num_rx, num_tx, max_num_paths)
        self.theta_r = np.expand_dims(self.theta_r, axis=0)
        # (1, num_rx, num_tx, max_num_paths)
        self.phi_r = np.expand_dims(self.phi_r, axis=0)
        # (1, max_num_paths)
        self.types = np.expand_dims(self.types, axis=0)
        # (1, max_num_paths)
        self.doppler = np.expand_dims(self.doppler, axis=0)

        tau = self.tau
        if tau.shape[-1] == 0:  # No paths
            self._min_tau = np.zeros_like(tau)
        else:
            tau = np.where(tau < 0, np.inf, tau)
            # (1, num_rx, num_tx, 1)
            min_tau = np.min(tau, axis=3, keepdims=True)
            min_tau = np.where(min_tau == np.inf, 0.0, min_tau)
            self._min_tau = min_tau

        # Add the time steps dimension
        # (1, num_rx, num_tx, max_num_paths, 1)
        self.a = np.expand_dims(self.a, axis=-1)

        # Normalize delays
        self.normalize_delays = True

    def set_los_path_type(self):
        """
        Flags paths that do not hit any object as LoS
        """

        # (max_depth, num_targets, num_sources, num_paths)
        objects = self.objects
        # (num_targets, num_sources, num_paths)
        mask = self.targets_sources_mask

        if objects.shape[3] > 0:
            # (num_targets, num_sources, num_paths)
            los_path = np.all(objects == -1, axis=0)
            # (num_targets, num_sources, num_paths)
            los_path = np.logical_and(los_path, mask)
            # (num_paths,)
            los_path = np.any(los_path, axis=(0, 1))
            # (1,)
            los_path_index = np.where(los_path)
            updates = np.repeat(Paths.LOS, los_path_index.shape[0], 0)
            self.types[los_path_index] = updates

    def pad_or_crop(self, a, tau, k):
        """
        Enforces that CIRs have exactly k paths by either
        zero-padding of cropping the weakest paths
        """
        max_num_paths = a.shape[-2]

        # Crop
        if k < max_num_paths:
            # Compute indices of the k strongest paths
            # As is independent of the number of time steps,
            # Therefore, we use only the first one a[...,0].
            # ind : (num_rx, num_tx, k)
            ind = np.argsort(np.abs(a[..., 0]), axis=-1)[..., -k:]

            # Gather the strongest paths
            # (num_rx, num_tx, k, num_time_steps)
            a = a[ind, :]

            # tau : (num_rx, num_tx, max_num_paths)

            # Get relevant delays
            # (num_rx, num_tx, k)
            tau = tau[ind]

        # Pad
        elif k > max_num_paths:
            # Pad paths with zeros
            pad_size = k - max_num_paths

            # Paddings for the paths gains
            a = np.pad(
                a,
                [[0, 0], [0, 0], [0, pad_size], [0, 0]],
                "constant",
                constant_values=0,
            )

            # Paddings for the delays (-1 by Sionna convention)
            tau = np.pad(
                tau, [[0, 0], [0, 0], [0, pad_size]], "constant", constant_values=-1
            )

        return a, tau
