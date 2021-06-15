from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import properties
from scipy.constants import mu_0
import numpy as np
from scipy import sparse as sp
from scipy.linalg import block_diag

from ..data import Data
from ..maps import IdentityMap
from ..simulation import BaseSimulation
from ..survey import BaseSurvey, BaseSrc

# from ..frequency_domain.survey import FDSurvey
# from ..time_domain.survey import TDSurvey
from .. import utils
from ..utils import sdiag, Zero, mkvc
from .. import props
from empymod.utils import check_hankel

try:
    from multiprocessing import Pool
    from sys import platform
except ImportError:
    print("multiprocessing is not available")
    PARALLEL = False
else:
    PARALLEL = True
    import multiprocessing

__all__ = ["BaseEM1DSimulation", "BaseStitchedEM1DSimulation"]


###############################################################################
#                                                                             #
#                             Base EM1D Simulation                            #
#                                                                             #
###############################################################################


class BaseEM1DSimulation(BaseSimulation):
    """
    Base simulation class for simulating the EM response over a 1D layered Earth
    for a single sounding. The simulation computes the fields by solving the
    Hankel transform solutions from Electromagnetic Theory for Geophysical
    Applications: Chapter 4 (Ward and Hohmann, 1988).
    """

    hankel_filter = "key_101_2009"  # Default: Hankel filter
    hankel_pts_per_dec = None  # Default: Standard DLF
    verbose = False
    fix_Jmatrix = False
    _Jmatrix_sigma = None
    _Jmatrix_height = None
    _pred = None
    use_sounding = False  # Default: False (not optimized)
    _formulation = "1D"

    # Properties for electrical conductivity/resistivity
    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )

    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    eta = props.PhysicalProperty(
        "Intrinsic chargeability (V/V), 0 <= eta < 1", default=0.0
    )
    tau = props.PhysicalProperty("Time constant for Cole-Cole model (s)", default=1.0)
    c = props.PhysicalProperty(
        "Frequency Dependency for Cole-Cole model, 0 < c < 1", default=0.5
    )
    # eta, etaMap, etaDeriv = props.Invertible(
    #     "Intrinsic chargeability (V/V), 0 <= eta < 1", default=0.0
    # )
    #
    # tau, tauMap, tauDeriv = props.Invertible(
    #     "Time constant for Cole-Cole model (s)", default=1.0
    # )
    #
    # c, cMap, cDeriv = props.Invertible(
    #     "Frequency Dependency for Cole-Cole model, 0 < c < 1", default=0.5
    # )

    # Properties for magnetic susceptibility
    mu, muMap, muDeriv = props.Invertible(
        "Magnetic permeability at infinite frequency (SI)", default=mu_0
    )
    dchi = props.PhysicalProperty(
        "DC magnetic susceptibility for viscous remanent magnetization contribution (SI)",
        default=0.0,
    )
    tau1 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)",
        default=1e-10,
    )
    tau2 = props.PhysicalProperty(
        "Upper bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)",
        default=10.0,
    )

    # dchi, dchiMap, dchiDeriv = props.Invertible(
    #     "DC magnetic susceptibility for viscous remanent magnetization contribution (SI)",
    #     default=0.0,
    # )
    #
    # tau1, tau1Map, tau1Deriv = props.Invertible(
    #     "Lower bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)",
    #     default=1e-10,
    # )
    #
    # tau2, tau2Map, tau2Deriv = props.Invertible(
    #     "Upper bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)",
    #     default=10.0,
    # )

    # Additional properties
    h, hMap, hDeriv = props.Invertible("Receiver Height (m), h > 0",)

    survey = properties.Instance("a survey object", BaseSurvey, required=True)

    topo = properties.Array("Topography (x, y, z)", dtype=float)

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "layer thicknesses (m)", default=np.array([])
    )

    def __init__(self, **kwargs):
        BaseSimulation.__init__(self, **kwargs)

        # Check input arguments. If self.hankel_filter is not a valid filter,
        # it will set it to the default (key_201_2009).
        ht, htarg = check_hankel(
            "dlf", {"dlf": self.hankel_filter, "pts_per_dec": 0}, 1
        )

        self.fhtfilt = htarg["dlf"]  # Store filter
        self.hankel_pts_per_dec = htarg["pts_per_dec"]  # Store pts_per_dec
        if self.verbose:
            print(">> Use " + self.hankel_filter + " filter for Hankel Transform")

    @property
    def halfspace_switch(self):
        """True = halfspace, False = layered Earth"""
        if (self.thicknesses is None) | (len(self.thicknesses) == 0):
            return True
        else:
            return False

    @property
    def n_layer(self):
        """number of layers"""
        if self.halfspace_switch is False:
            return int(self.thicknesses.size + 1)
        elif self.halfspace_switch is True:
            return int(1)

    @property
    def n_filter(self):
        """ Length of filter """
        return self.fhtfilt.base.size

    @property
    def depth(self):
        """layer depths"""
        if self.thicknesses is not None:
            return np.r_[0.0, -np.cumsum(self.thicknesses)]

    def compute_sigma_matrix(self, frequencies):
        """
        Computes the complex conductivity matrix using Pelton's Cole-Cole model:

        .. math ::
            \\sigma (\\omega ) = \\sigma \\Bigg [
            1 - \\eta \\Bigg ( \\frac{1}{1 + (1-\\eta ) (1 + i\\omega \\tau)^c} \\Bigg )
            \\Bigg ]

        :param numpy.array frequencies: np.array(N,) containing frequencies
        :rtype: numpy.ndarray: np.array(n_layer, n_frequency)
        :return: complex conductivity matrix

        """
        n_layer = self.n_layer
        n_frequency = len(frequencies)
        # n_filter = self.n_filter

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))

        # No IP effect
        if np.all(self.eta) == 0.0:
            return sigma

        # IP effect
        else:

            if np.isscalar(self.eta):
                eta = self.eta
                tau = self.tau
                c = self.c
            else:
                eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
                tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
                c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

            w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

            sigma_complex = np.empty(
                [n_layer, n_frequency], dtype=np.complex128, order="F"
            )
            sigma_complex[:, :] = sigma - sigma * eta / (
                1 + (1 - eta) * (1j * w * tau) ** c
            )

            return sigma_complex

    def compute_mu_matrix(self, frequencies):
        """
        Computes the complex magnetic permeability matrix assuming a log-uniform
        distribution of time-relaxation constants:

        .. math::
            \\chi (\\omega ) = \\chi + \\Delta \\chi \\Bigg [
            1 - \\Bigg ( \\frac{1}{ln (\\tau_2 / \\tau_1 )} \\Bigg )
            ln \\Bigg ( \\frac{1 + i\\omega \\tau_2}{1 + i\\omega tau_1} ) \\Bigg )
            \\Bigg ]

        :param numpy.array frequencies: np.array(N,) containing frequencies
        :rtype: numpy.ndarray: np.array(n_layer, n_frequency)
        :return: complex magnetic susceptibility matrix
        """

        if np.isscalar(self.mu):
            mu = np.ones_like(self.sigma) * self.mu
        else:
            mu = self.mu

        n_layer = self.n_layer
        n_frequency = len(frequencies)
        # n_filter = self.n_filter

        mu = np.tile(mu.reshape([-1, 1]), (1, n_frequency))

        # No magnetic viscosity
        if np.all(self.dchi) == 0.0:

            return mu

        # Magnetic viscosity
        else:

            if np.isscalar(self.dchi):
                dchi = self.dchi * np.ones_like(self.chi)
                tau1 = self.tau1 * np.ones_like(self.chi)
                tau2 = self.tau2 * np.ones_like(self.chi)
            else:
                dchi = np.tile(self.dchi.reshape([-1, 1]), (1, n_frequency))
                tau1 = np.tile(self.tau1.reshape([-1, 1]), (1, n_frequency))
                tau2 = np.tile(self.tau2.reshape([-1, 1]), (1, n_frequency))

            w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

            mu_complex = mu + mu_0 * dchi * (
                1
                - np.log((1 + 1j * w * tau2) / (1 + 1j * w * tau1))
                / np.log(tau2 / tau1)
            )

            return mu_complex

    def compute_chi_matrix(self, frequencies):
        """
        Computes the complex magnetic permeability matrix assuming a log-uniform
        distribution of time-relaxation constants:

        .. math::
            \\chi (\\omega ) = \\chi + \\Delta \\chi \\Bigg [
            1 - \\Bigg ( \\frac{1}{ln (\\tau_2 / \\tau_1 )} \\Bigg )
            ln \\Bigg ( \\frac{1 + i\\omega \\tau_2}{1 + i\\omega tau_1} ) \\Bigg )
            \\Bigg ]

        :param numpy.array frequencies: np.array(N,) containing frequencies
        :rtype: numpy.ndarray: np.array(n_layer, n_frequency)
        :return: complex magnetic susceptibility matrix
        """

        if np.isscalar(self.mu):
            mu = np.ones_like(self.sigma) * self.mu
        else:
            mu = self.mu

        n_layer = self.n_layer
        n_frequency = len(frequencies)
        # n_filter = self.n_filter

        mu = np.tile(mu.reshape([-1, 1]), (1, n_frequency))

        # No magnetic viscosity
        if np.all(self.dchi) == 0.0:

            return mu

        # Magnetic viscosity
        else:

            if np.isscalar(self.dchi):
                dchi = self.dchi * np.ones_like(self.chi)
                tau1 = self.tau1 * np.ones_like(self.chi)
                tau2 = self.tau2 * np.ones_like(self.chi)
            else:
                dchi = np.tile(self.dchi.reshape([-1, 1]), (1, n_frequency))
                tau1 = np.tile(self.tau1.reshape([-1, 1]), (1, n_frequency))
                tau2 = np.tile(self.tau2.reshape([-1, 1]), (1, n_frequency))

            w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

            chi_complex = mu / mu_0 + dchi * (
                1
                - np.log((1 + 1j * w * tau2) / (1 + 1j * w * tau1))
                / np.log(tau2 / tau1)
            )

            return chi_complex

    def fields(self, m):
        if self.use_sounding:
            data = self.compute_integral_by_sounding(m, output_type="response")
            return data.dobs
        else:
            f = self.compute_integral(m, output_type="response")
            f = self.project_fields(f, output_type="response")
            return np.hstack(f)

    def dpred(self, m, f=None):
        """
        Computes predicted data.
        Here we do not store predicted data
        because projection (`d = P(f)`) is cheap.
        """

        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)

        return f

    def getJ_height(self, m, f=None):
        """
        Compute the sensitivity with respect to source height(s).
        """

        # Null if source height is not parameter of the simulation.
        if self.hMap is None:
            return utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height

        else:

            if self.verbose:
                print(">> Compute J height ")
            if self.use_sounding:
                dudh = self.compute_integral_by_sounding(
                    m, output_type="sensitivity_height"
                )
                self._Jmatrix_height = dudh.dobs.reshape([-1, 1])
            else:
                dudh = self.compute_integral(m, output_type="sensitivity_height")
                self._Jmatrix_height = np.hstack(
                    self.project_fields(dudh, output_type="sensitivity_height")
                )
                self._Jmatrix_height = np.hstack(dudh).reshape([-1, 1])
            return self._Jmatrix_height

    def getJ_sigma(self, m, f=None):
        """
        Compute the sensitivity with respect to static conductivity.
        """

        # Null if sigma is not parameter of the simulation.
        if self.sigmaMap is None:
            return utils.Zero()

        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        else:

            if self.verbose:
                print(">> Compute J sigma")

            if self.use_sounding:
                dudsig = self.compute_integral_by_sounding(
                    m, output_type="sensitivity_sigma"
                )
                self._Jmatrix_sigma = dudsig.sensitivity
            else:
                dudsig = self.compute_integral(m, output_type="sensitivity_sigma")
                self._Jmatrix_sigma = np.vstack(
                    self.project_fields(dudsig, output_type="sensitivity_sigma")
                )

            if self._Jmatrix_sigma.ndim == 1:
                self._Jmatrix_sigma = self._Jmatrix_sigma.reshape([-1, 1])
            return self._Jmatrix_sigma

    def getJ(self, m, f=None):
        """
        Fetch Jacobian.
        """
        return (
            self.getJ_sigma(m, f=f) * self.sigmaDeriv
            + self.getJ_height(m, f=f) * self.hDeriv
        )

    def Jvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jv = np.dot(J_sigma, self.sigmaMap.deriv(m, v))
        if self.hMap is not None:
            Jv += np.dot(J_height, self.hMap.deriv(m, v))
        return Jv

    def Jtvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jtv = self.sigmaDeriv.T * np.dot(J_sigma.T, v)
        if self.hMap is not None:
            Jtv += self.hDeriv.T * np.dot(J_height.T, v)
        return Jtv

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ["_Jmatrix_sigma"]
            if self._Jmatrix_height is not None:
                toDelete += ["_Jmatrix_height"]
        return toDelete

    def depth_of_investigation_christiansen_2012(self, std, thres_hold=0.8):
        pred = self.survey._pred.copy()
        delta_d = std * np.log(abs(self.survey.dobs))
        J = self.getJ(self.model)
        J_sum = abs(utils.sdiag(1 / delta_d / pred) * J).sum(axis=0)
        S = np.cumsum(J_sum[::-1])[::-1]
        active = S - thres_hold > 0.0
        doi = abs(self.depth[active]).max()
        return doi, active

    def get_threshold(self, uncert):
        _, active = self.depth_of_investigation(uncert)
        JtJdiag = self.get_JtJdiag(uncert)
        delta = JtJdiag[active].min()
        return delta

    def get_JtJdiag(self, uncert):
        J = self.getJ(self.model)
        JtJdiag = (np.power((utils.sdiag(1.0 / uncert) * J), 2)).sum(axis=0)
        return JtJdiag


class BaseStitchedEM1DSimulation(BaseSimulation):
    """
    Base class for the stitched 1D simulation. This simulation models the EM
    response for a set of 1D EM soundings.
    """

    _Jmatrix_sigma = None
    _Jmatrix_height = None
    run_simulation = None
    n_cpu = None
    parallel = False
    parallel_jvec_jtvec = False
    verbose = False
    fix_Jmatrix = False
    invert_height = None
    n_sounding_for_chunk = None
    use_sounding = True

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers", default=np.array([])
    )

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")

    h, hMap, hDeriv = props.Invertible("Receiver Height (m), h > 0",)

    eta = props.PhysicalProperty("Electrical chargeability (V/V), 0 <= eta < 1")

    tau = props.PhysicalProperty("Time constant (s)")

    c = props.PhysicalProperty("Frequency Dependency, 0 < c < 1")

    chi = props.PhysicalProperty("Magnetic susceptibility (SI)")

    dchi = props.PhysicalProperty(
        "DC magnetic susceptibility attributed to magnetic viscosity (SI)"
    )

    tau1 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants (s)"
    )

    tau2 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants (s)"
    )

    topo = properties.Array("Topography (x, y, z)", dtype=float, shape=("*", 3))

    survey = properties.Instance("a survey object", BaseSurvey, required=True)

    def __init__(self, **kwargs):
        utils.setKwargs(self, **kwargs)

        if PARALLEL:
            if self.parallel:
                print(">> Use multiprocessing for parallelization")
                if self.n_cpu is None:
                    self.n_cpu = multiprocessing.cpu_count()
                print((">> n_cpu: %i") % (self.n_cpu))
            else:
                print(">> Serial version is used")
        else:
            print(">> Serial version is used")

        if self.hMap is None:
            self.invert_height = False
        else:
            self.invert_height = True

    # ------------- For survey ------------- #
    # @property
    # def dz(self):
    #     if self.mesh.dim==2:
    #         return self.mesh.dy
    #     elif self.mesh.dim==3:
    #         return self.mesh.dz

    @property
    def halfspace_switch(self):
        """True = halfspace, False = layered Earth"""
        if (self.thicknesses is None) | (len(self.thicknesses) == 0):
            return True
        else:
            return False

    @property
    def n_layer(self):
        if self.thicknesses is None:
            return 1
        else:
            return len(self.thicknesses) + 1

    @property
    def n_sounding(self):
        return len(self.survey.source_location_by_sounding_dict)

    @property
    def data_index(self):
        return self.survey.data_index

    # ------------- For physical properties ------------- #
    @property
    def Sigma(self):
        if getattr(self, "_Sigma", None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    @property
    def Eta(self):
        if getattr(self, "_Eta", None) is None:
            # Ordering: first z then x
            if self.eta is None:
                self._Eta = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Eta = self.eta.reshape((self.n_sounding, self.n_layer))
        return self._Eta

    @property
    def Tau(self):
        if getattr(self, "_Tau", None) is None:
            # Ordering: first z then x
            if self.tau is None:
                self._Tau = 1e-3 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Tau = self.tau.reshape((self.n_sounding, self.n_layer))
        return self._Tau

    @property
    def C(self):
        if getattr(self, "_C", None) is None:
            # Ordering: first z then x
            if self.c is None:
                self._C = np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._C = self.c.reshape((self.n_sounding, self.n_layer))
        return self._C

    @property
    def Chi(self):
        if getattr(self, "_Chi", None) is None:
            # Ordering: first z then x
            if self.chi is None:
                self._Chi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Chi = self.chi.reshape((self.n_sounding, self.n_layer))
        return self._Chi

    @property
    def dChi(self):
        if getattr(self, "_dChi", None) is None:
            # Ordering: first z then x
            if self.dchi is None:
                self._dChi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._dChi = self.dchi.reshape((self.n_sounding, self.n_layer))
        return self._dChi

    @property
    def Tau1(self):
        if getattr(self, "_Tau1", None) is None:
            # Ordering: first z then x
            if self.tau1 is None:
                self._Tau1 = 1e-10 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Tau1 = self.tau1.reshape((self.n_sounding, self.n_layer))
        return self._Tau1

    @property
    def Tau2(self):
        if getattr(self, "_Tau2", None) is None:
            # Ordering: first z then x
            if self.tau2 is None:
                self._Tau2 = 100.0 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Tau2 = self.tau2.reshape((self.n_sounding, self.n_layer))
        return self._Tau2

    @property
    def JtJ_sigma(self):
        return self._JtJ_sigma

    def JtJ_height(self):
        return self._JtJ_height

    @property
    def H(self):
        if self.hMap is None:
            return np.ones(self.n_sounding)
        else:
            return self.h

    # ------------- Etcetra .... ------------- #
    @property
    def IJLayers(self):
        if getattr(self, "_IJLayers", None) is None:
            # Ordering: first z then x
            self._IJLayers = self.set_ij_n_layer()
        return self._IJLayers

    @property
    def IJHeight(self):
        if getattr(self, "_IJHeight", None) is None:
            # Ordering: first z then x
            self._IJHeight = self.set_ij_n_layer(n_layer=1)
        return self._IJHeight

    # ------------- For physics ------------- #

    def input_args(self, i_sounding, output_type="forward"):
        output = (
            self.survey.get_sources_by_sounding_number(i_sounding),
            self.topo[i_sounding, :],
            self.thicknesses,
            self.Sigma[i_sounding, :],
            self.Eta[i_sounding, :],
            self.Tau[i_sounding, :],
            self.C[i_sounding, :],
            self.Chi[i_sounding, :],
            self.dChi[i_sounding, :],
            self.Tau1[i_sounding, :],
            self.Tau2[i_sounding, :],
            self.H[i_sounding],
            output_type,
            self.invert_height,
        )
        return output

    def fields(self, m):
        if self.verbose:
            print("Compute fields")

        return self.forward(m)

    def dpred(self, m, f=None):
        """
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.fields is called.
        """
        if f is None:
            f = self.fields(m)

        return f

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        # Set flat topo at zero
        if self.topo is None:
            self.set_null_topography()

        run_simulation = self.run_simulation

        if self.parallel:
            if self.verbose:
                print("parallel")
            pool = Pool(self.n_cpu)

            # This assumes the same # of layers for each of sounding
            if self.n_sounding_for_chunk is None:
                result = pool.map(
                    run_simulation,
                    [
                        self.input_args(i, output_type="forward")
                        for i in range(self.n_sounding)
                    ],
                )
            else:
                result = pool.map(
                    self._run_simulation_by_chunk,
                    [
                        self.input_args_by_chunk(i, output_type="forward")
                        for i in range(self.n_chunk)
                    ],
                )
                return np.r_[result].ravel()

            pool.close()
            pool.join()
        else:
            result = [
                run_simulation(self.input_args(i, output_type="forward"))
                for i in range(self.n_sounding)
            ]
        return np.hstack(result)

    @property
    def sounding_number(self):
        self._sounding_number = [
            key for key in self.survey.source_location_by_sounding_dict.keys()
        ]
        return self._sounding_number

    @property
    def sounding_number_chunks(self):
        self._sounding_number_chunks = list(
            self.chunks(self.sounding_number, self.n_sounding_for_chunk)
        )
        return self._sounding_number_chunks

    @property
    def n_chunk(self):
        self._n_chunk = len(self.sounding_number_chunks)
        return self._n_chunk

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def input_args_by_chunk(self, i_chunk, output_type):
        args_by_chunks = []
        for i_sounding in self.sounding_number_chunks[i_chunk]:
            args_by_chunks.append(self.input_args(i_sounding, output_type))
        return args_by_chunks

    def set_null_topography(self):
        self.topo = np.vstack(
            [
                np.c_[src.location[0], src.location[1], 0.0]
                for i, src in enumerate(self.survey.source_list)
            ]
        )

    def set_ij_n_layer(self, n_layer=None):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        if n_layer is None:
            m = self.n_layer
        else:
            m = n_layer
        source_location_by_sounding_dict = self.survey.source_location_by_sounding_dict
        for i_sounding in range(self.n_sounding):
            n = self.survey.vnD_by_sounding_dict[i_sounding]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order="F") + shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)

    def set_ij_height(self):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        m = self.n_layer
        for i_sounding in range(self.n_sounding):
            n = self.survey.vnD_by_sounding_dict[i_sounding]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order="F") + shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)

    def getJ_sigma(self, m):
        """
             Compute d F / d sigma
        """
        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        if self.verbose:
            print(">> Compute J sigma")
        self.model = m

        run_simulation = self.run_simulation

        if self.parallel:

            pool = Pool(self.n_cpu)
            if self.n_sounding_for_chunk is None:
                self._Jmatrix_sigma = pool.map(
                    run_simulation,
                    [
                        self.input_args(i, output_type="sensitivity_sigma")
                        for i in range(self.n_sounding)
                    ],
                )
                self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
            else:
                self._Jmatrix_sigma = pool.map(
                    self._run_simulation_by_chunk,
                    [
                        self.input_args_by_chunk(i, output_type="sensitivity_sigma")
                        for i in range(self.n_chunk)
                    ],
                )
                self._Jmatrix_sigma = np.r_[self._Jmatrix_sigma].ravel()

            pool.close()
            pool.join()

            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()

        else:
            self._Jmatrix_sigma = [
                run_simulation(self.input_args(i, output_type="sensitivity_sigma"))
                for i in range(self.n_sounding)
            ]
            self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()

        return self._Jmatrix_sigma

    def getJ_height(self, m):
        """
             Compute d F / d height
        """
        if self.hMap is None:
            return utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        if self.verbose:
            print(">> Compute J height")

        self.model = m

        run_simulation = self.run_simulation

        if (self.parallel) & (__name__ == "__main__"):
            pool = Pool(self.n_cpu)
            if self.n_sounding_for_chunk is None:
                self._Jmatrix_height = pool.map(
                    run_simulation,
                    [
                        self.input_args(i, output_type="sensitivity_height")
                        for i in range(self.n_sounding)
                    ],
                )
            else:
                self._Jmatrix_height = pool.map(
                    self._run_simulation_by_chunk,
                    [
                        self.input_args_by_chunk(i, output_type="sensitivity_height")
                        for i in range(self.n_chunk)
                    ],
                )
            pool.close()
            pool.join()
            if self.parallel_jvec_jtvec is False:
                # self._Jmatrix_height = sp.block_diag(self._Jmatrix_height).tocsr()
                self._Jmatrix_height = np.hstack(self._Jmatrix_height)
                self._Jmatrix_height = sp.coo_matrix(
                    (self._Jmatrix_height, self.IJHeight), dtype=float
                ).tocsr()
        else:
            self._Jmatrix_height = [
                run_simulation(self.input_args(i, output_type="sensitivity_height"))
                for i in range(self.n_sounding)
            ]
            self._Jmatrix_height = np.hstack(self._Jmatrix_height)
            self._Jmatrix_height = sp.coo_matrix(
                (self._Jmatrix_height, self.IJHeight), dtype=float
            ).tocsr()

        return self._Jmatrix_height

    def Jvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        Jv = J_sigma * (utils.sdiag(1.0 / self.sigma) * (self.sigmaDeriv * v))
        if self.hMap is not None:
            Jv += J_height * (self.hDeriv * v)
        return Jv

    def Jtvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)

        Jtv = self.sigmaDeriv.T * (utils.sdiag(1.0 / self.sigma) * (J_sigma.T * v))
        if self.hMap is not None:
            Jtv += self.hDeriv.T * (J_height.T * v)
        return Jtv

    def getJtJdiag(self, m, W=None, threshold=1e-8):
        """
        Compute diagonal component of JtJ or
        trace of sensitivity matrix (J)
        """
        J_sigma = self.getJ_sigma(m)
        J_matrix = J_sigma * (utils.sdiag(1.0 / self.sigma) * (self.sigmaDeriv))

        if self.hMap is not None:
            J_height = self.getJ_height(m)
            J_matrix += J_height * self.hDeriv

        if W is None:
            W = utils.speye(J_matrix.shape[0])

        J_matrix = W * J_matrix
        JtJ_diag = (J_matrix.T * J_matrix).diagonal()
        JtJ_diag /= JtJ_diag.max()
        JtJ_diag += threshold
        return JtJ_diag

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None:
            toDelete += ["_Sigma"]
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ["_Jmatrix_sigma"]
            if self._Jmatrix_height is not None:
                toDelete += ["_Jmatrix_height"]
        return toDelete

    def _run_simulation_by_chunk(self, args_chunk):
        """
        This method simulates the EM response or computes the sensitivities for
        a single sounding. The method allows for parallelization of
        the stitched 1D problem.
        """
        n = len(args_chunk)
        results = [
            self.run_simulation(args_chunk[i_sounding]) for i_sounding in range(n)
        ]
        return results


class Sensitivity(Data):

    sensitivity = properties.Array(
        """
        Matrix of the sensitivity.
        parameters:

        .. code:: python

            sensitivity = Data(survey)
            for src in survey.source_list:
                for rx in src.receiver_list:
                    sensitivity[src, rx] = sensitivity_for_a_datum

        """,
        shape=("*", "*"),
        required=True,
    )

    M = properties.Integer(
        """
        """,
        required=True,
    )

    #######################
    # Instantiate the class
    #######################
    def __init__(
        self, survey, sensitivity=None, M=None,
    ):
        super(Data, self).__init__()
        self.survey = survey
        self.M = M

        # Observed data
        if sensitivity is None:
            sensitivity = np.nan * np.ones((survey.nD, M))  # initialize data as nans
        self.sensitivity = sensitivity

    @properties.validator("sensitivity")
    def _sensitivity_validator(self, change):
        if change["value"].shape != (self.survey.nD, self.M):
            raise ValueError()

    ##########################
    # Methods
    ##########################

    def __setitem__(self, key, value):
        index = self.index_dictionary[key[0]][key[1]]
        self.sensitivity[index, :] = value

    def __getitem__(self, key):
        index = self.index_dictionary[key[0]][key[1]]
        return self.sensitivity[index, :]