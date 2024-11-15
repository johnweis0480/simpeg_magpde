import warnings
import numpy as np
import scipy.sparse as sp
from geoana.kernels import (
    prism_fxxy,
    prism_fxxz,
    prism_fxyz,
    prism_fzx,
    prism_fzy,
    prism_fzz,
    prism_fzzz,
)
from scipy.constants import mu_0

from simpeg import props, utils
from simpeg.utils import mat_utils, mkvc, sdiag
from simpeg.utils.code_utils import deprecate_property, validate_string, validate_type
from simpeg.utils.solver_utils import get_default_solver

from ...base import BaseMagneticPDESimulation
from ..base import BaseEquivalentSourceLayerSimulation, BasePFSimulation
from .analytics import CongruousMagBC
from .survey import Survey

from ._numba_functions import (
    choclo,
    _sensitivity_tmi_parallel,
    _sensitivity_tmi_serial,
    _sensitivity_mag_parallel,
    _sensitivity_mag_serial,
    _forward_tmi_parallel,
    _forward_tmi_serial,
    _forward_mag_parallel,
    _forward_mag_serial,
    _forward_tmi_2d_mesh_serial,
    _forward_tmi_2d_mesh_parallel,
    _forward_mag_2d_mesh_serial,
    _forward_mag_2d_mesh_parallel,
    _forward_tmi_derivative_2d_mesh_serial,
    _forward_tmi_derivative_2d_mesh_parallel,
    _sensitivity_mag_2d_mesh_serial,
    _sensitivity_mag_2d_mesh_parallel,
    _sensitivity_tmi_2d_mesh_serial,
    _sensitivity_tmi_2d_mesh_parallel,
    _forward_tmi_derivative_parallel,
    _forward_tmi_derivative_serial,
    _sensitivity_tmi_derivative_parallel,
    _sensitivity_tmi_derivative_serial,
    _sensitivity_tmi_derivative_2d_mesh_serial,
    _sensitivity_tmi_derivative_2d_mesh_parallel,
)

if choclo is not None:
    CHOCLO_SUPPORTED_COMPONENTS = {
        "tmi",
        "bx",
        "by",
        "bz",
        "bxx",
        "byy",
        "bzz",
        "bxy",
        "bxz",
        "byz",
        "tmi_x",
        "tmi_y",
        "tmi_z",
    }
    CHOCLO_KERNELS = {
        "bx": (choclo.prism.kernel_ee, choclo.prism.kernel_en, choclo.prism.kernel_eu),
        "by": (choclo.prism.kernel_en, choclo.prism.kernel_nn, choclo.prism.kernel_nu),
        "bz": (choclo.prism.kernel_eu, choclo.prism.kernel_nu, choclo.prism.kernel_uu),
        "bxx": (
            choclo.prism.kernel_eee,
            choclo.prism.kernel_een,
            choclo.prism.kernel_eeu,
        ),
        "byy": (
            choclo.prism.kernel_enn,
            choclo.prism.kernel_nnn,
            choclo.prism.kernel_nnu,
        ),
        "bzz": (
            choclo.prism.kernel_euu,
            choclo.prism.kernel_nuu,
            choclo.prism.kernel_uuu,
        ),
        "bxy": (
            choclo.prism.kernel_een,
            choclo.prism.kernel_enn,
            choclo.prism.kernel_enu,
        ),
        "bxz": (
            choclo.prism.kernel_eeu,
            choclo.prism.kernel_enu,
            choclo.prism.kernel_euu,
        ),
        "byz": (
            choclo.prism.kernel_enu,
            choclo.prism.kernel_nnu,
            choclo.prism.kernel_nuu,
        ),
        "tmi_x": (
            choclo.prism.kernel_eee,
            choclo.prism.kernel_enn,
            choclo.prism.kernel_euu,
            choclo.prism.kernel_een,
            choclo.prism.kernel_eeu,
            choclo.prism.kernel_enu,
        ),
        "tmi_y": (
            choclo.prism.kernel_een,
            choclo.prism.kernel_nnn,
            choclo.prism.kernel_nuu,
            choclo.prism.kernel_enn,
            choclo.prism.kernel_enu,
            choclo.prism.kernel_nnu,
        ),
        "tmi_z": (
            choclo.prism.kernel_eeu,
            choclo.prism.kernel_nnu,
            choclo.prism.kernel_uuu,
            choclo.prism.kernel_enu,
            choclo.prism.kernel_euu,
            choclo.prism.kernel_nuu,
        ),
    }
    CHOCLO_FORWARD_FUNCS = {
        "bx": choclo.prism.magnetic_e,
        "by": choclo.prism.magnetic_n,
        "bz": choclo.prism.magnetic_u,
        "bxx": choclo.prism.magnetic_ee,
        "byy": choclo.prism.magnetic_nn,
        "bzz": choclo.prism.magnetic_uu,
        "bxy": choclo.prism.magnetic_en,
        "bxz": choclo.prism.magnetic_eu,
        "byz": choclo.prism.magnetic_nu,
    }


class Simulation3DIntegral(BasePFSimulation):
    """
    Magnetic simulation in integral form.

    Parameters
    ----------
    mesh : discretize.TreeMesh or discretize.TensorMesh
        Mesh use to run the magnetic simulation.
    survey : simpeg.potential_fields.magnetics.Survey
        Magnetic survey with information of the receivers.
    active_cells : (n_cells) numpy.ndarray, optional
        Array that indicates which cells in ``mesh`` are active cells.
    chi : numpy.ndarray, optional
        Susceptibility array for the active cells in the mesh.
    chiMap : Mapping, optional
        Model mapping.
    model_type : str, optional
        Whether the model are susceptibilities of the cells (``"scalar"``),
        or effective susceptibilities (``"vector"``).
    is_amplitude_data : bool, optional
        If True, the returned fields will be the amplitude of the magnetic
        field. If False, the fields will be returned unmodified.
    sensitivity_dtype : numpy.dtype, optional
        Data type that will be used to build the sensitivity matrix.
    store_sensitivities : {"ram", "disk", "forward_only"}
        Options for storing sensitivity matrix. There are 3 options

        - 'ram': sensitivities are stored in the computer's RAM
        - 'disk': sensitivities are written to a directory
        - 'forward_only': you intend only do perform a forward simulation and
          sensitivities do not need to be stored

    sensitivity_path : str, optional
        Path to store the sensitivity matrix if ``store_sensitivities`` is set
        to ``"disk"``. Default to "./sensitivities".
    engine : {"geoana", "choclo"}, optional
       Choose which engine should be used to run the forward model.
    numba_parallel : bool, optional
        If True, the simulation will run in parallel. If False, it will
        run in serial. If ``engine`` is not ``"choclo"`` this argument will be
        ignored.
    ind_active : np.ndarray of int or bool

        .. deprecated:: 0.23.0

           Argument ``ind_active`` is deprecated in favor of
           ``active_cells`` and will be removed in SimPEG v0.24.0.
    """

    chi, chiMap, chiDeriv = props.Invertible("Magnetic Susceptibility (SI)")

    def __init__(
        self,
        mesh,
        chi=None,
        chiMap=None,
        model_type="scalar",
        is_amplitude_data=False,
        engine="geoana",
        numba_parallel=True,
        **kwargs,
    ):
        self.model_type = model_type
        super().__init__(mesh, engine=engine, numba_parallel=numba_parallel, **kwargs)
        self.chi = chi
        self.chiMap = chiMap

        self._G = None
        self._M = None
        self._gtg_diagonal = None
        self.is_amplitude_data = is_amplitude_data
        self.modelMap = self.chiMap

        # Warn if n_processes has been passed
        if self.engine == "choclo" and "n_processes" in kwargs:
            warnings.warn(
                "The 'n_processes' will be ignored when selecting 'choclo' as the "
                "engine in the magnetic simulation.",
                UserWarning,
                stacklevel=1,
            )
            self.n_processes = None

        if self.engine == "choclo":
            if self.numba_parallel:
                self._sensitivity_tmi = _sensitivity_tmi_parallel
                self._sensitivity_mag = _sensitivity_mag_parallel
                self._forward_tmi = _forward_tmi_parallel
                self._forward_mag = _forward_mag_parallel
                self._forward_tmi_derivative = _forward_tmi_derivative_parallel
                self._sensitivity_tmi_derivative = _sensitivity_tmi_derivative_parallel
            else:
                self._sensitivity_tmi = _sensitivity_tmi_serial
                self._sensitivity_mag = _sensitivity_mag_serial
                self._forward_tmi = _forward_tmi_serial
                self._forward_mag = _forward_mag_serial
                self._forward_tmi_derivative = _forward_tmi_derivative_serial
                self._sensitivity_tmi_derivative = _sensitivity_tmi_derivative_serial

    @property
    def model_type(self):
        """Type of magnetization model

        Returns
        -------
        str
            A string defining the model type for the simulation.
            One of {'scalar', 'vector'}.
        """
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        self._model_type = validate_string("model_type", value, ["scalar", "vector"])

    @property
    def is_amplitude_data(self):
        return self._is_amplitude_data

    @is_amplitude_data.setter
    def is_amplitude_data(self, value):
        self._is_amplitude_data = validate_type("is_amplitude_data", value, bool)

    @property
    def M(self):
        """
        M: ndarray
            Magnetization matrix
        """
        if self.model_type == "vector":
            return None
        if getattr(self, "_M", None) is None:
            mag = self.survey.source_field.b0
            self._M = np.ones((self.nC, 3)) * mag
        return self._M

    @M.setter
    def M(self, M):
        """
        Create magnetization matrix from unit vector orientation
        :parameter
        M: array (3*nC,) or (nC, 3)
        """
        if self.model_type == "vector":
            self._M = sdiag(mkvc(M) * self.survey.source_field.amplitude)
        else:
            M = np.asarray(M)
            self._M = M.reshape((self.nC, 3))

    def fields(self, model):
        self.model = model
        # model = self.chiMap * model
        if self.store_sensitivities == "forward_only":
            if self.engine == "choclo":
                fields = self._forward(self.chi)
            else:
                fields = mkvc(self.linear_operator())
        else:
            fields = np.asarray(
                self.G @ self.chi.astype(self.sensitivity_dtype, copy=False)
            )

        if self.is_amplitude_data:
            fields = self.compute_amplitude(fields)

        return fields

    @property
    def G(self):
        if getattr(self, "_G", None) is None:
            if self.engine == "choclo":
                self._G = self._sensitivity_matrix()
            else:
                self._G = self.linear_operator()

        return self._G

    modelType = deprecate_property(
        model_type, "modelType", "model_type", removal_version="0.18.0", error=True
    )

    @property
    def nD(self):
        """
        Number of data
        """
        self._nD = self.survey.receiver_locations.shape[0]

        return self._nD

    @property
    def tmi_projection(self):
        if getattr(self, "_tmi_projection", None) is None:
            # Convert from north to cartesian
            self._tmi_projection = mat_utils.dip_azimuth2cartesian(
                self.survey.source_field.inclination,
                self.survey.source_field.declination,
            ).squeeze()

        return self._tmi_projection

    def getJtJdiag(self, m, W=None, f=None):
        """
        Return the diagonal of JtJ
        """
        self.model = m

        if W is None:
            W = np.ones(self.survey.nD)
        else:
            W = W.diagonal() ** 2
        if getattr(self, "_gtg_diagonal", None) is None:
            diag = np.zeros(self.G.shape[1])
            if not self.is_amplitude_data:
                for i in range(len(W)):
                    diag += W[i] * (self.G[i] * self.G[i])
            else:
                ampDeriv = self.ampDeriv
                Gx = self.G[::3]
                Gy = self.G[1::3]
                Gz = self.G[2::3]
                for i in range(len(W)):
                    row = (
                        ampDeriv[0, i] * Gx[i]
                        + ampDeriv[1, i] * Gy[i]
                        + ampDeriv[2, i] * Gz[i]
                    )
                    diag += W[i] * (row * row)
            self._gtg_diagonal = diag
        else:
            diag = self._gtg_diagonal
        return mkvc((sdiag(np.sqrt(diag)) @ self.chiDeriv).power(2).sum(axis=0))

    def Jvec(self, m, v, f=None):
        self.model = m
        dmu_dm_v = self.chiDeriv @ v

        Jvec = self.G @ dmu_dm_v.astype(self.sensitivity_dtype, copy=False)

        if self.is_amplitude_data:
            # dask doesn't support an "order" argument to reshape...
            Jvec = Jvec.reshape((-1, 3)).T  # reshape((3, -1), order="F")
            ampDeriv_Jvec = self.ampDeriv * Jvec
            return ampDeriv_Jvec[0] + ampDeriv_Jvec[1] + ampDeriv_Jvec[2]
        else:
            return Jvec

    def Jtvec(self, m, v, f=None):
        self.model = m

        if self.is_amplitude_data:
            v = self.ampDeriv * v
            # dask doesn't support and "order" argument to reshape...
            v = v.T.reshape(-1)  # .reshape(-1, order="F")
        Jtvec = self.G.T @ v.astype(self.sensitivity_dtype, copy=False)
        return np.asarray(self.chiDeriv.T @ Jtvec)

    @property
    def ampDeriv(self):
        if getattr(self, "_ampDeriv", None) is None:
            fields = np.asarray(
                self.G.dot(self.chi).astype(self.sensitivity_dtype, copy=False)
            )
            self._ampDeriv = self.normalized_fields(fields)

        return self._ampDeriv

    @classmethod
    def normalized_fields(cls, fields):
        """
        Return the normalized B fields
        """

        # Get field amplitude
        amp = cls.compute_amplitude(fields.astype(np.float64))

        return fields.reshape((3, -1), order="F") / amp[None, :]

    @classmethod
    def compute_amplitude(cls, b_xyz):
        """
        Compute amplitude of the magnetic field
        """

        amplitude = np.linalg.norm(b_xyz.reshape((3, -1), order="F"), axis=0)

        return amplitude

    def evaluate_integral(self, receiver_location, components):
        """
        Load in the active nodes of a tensor mesh and computes the magnetic
        forward relation between a cuboid and a given observation
        location outside the Earth [obsx, obsy, obsz]

        INPUT:
        receiver_location:  [obsx, obsy, obsz] nC x 3 Array

        components: list[str]
            List of magnetic components chosen from:
            'tmi', 'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz', 'tmi_x', 'tmi_y', 'tmi_z'

        OUTPUT:
        Tx = [Txx Txy Txz]
        Ty = [Tyx Tyy Tyz]
        Tz = [Tzx Tzy Tzz]
        """
        dr = self._nodes - receiver_location
        dx = dr[..., 0]
        dy = dr[..., 1]
        dz = dr[..., 2]

        node_evals = {}
        if "bx" in components or "tmi" in components:
            node_evals["gxx"] = prism_fzz(dy, dz, dx)
            node_evals["gxy"] = prism_fzx(dy, dz, dx)
            node_evals["gxz"] = prism_fzy(dy, dz, dx)
        if "by" in components or "tmi" in components:
            if "gxy" not in node_evals:
                node_evals["gxy"] = prism_fzx(dy, dz, dx)
            node_evals["gyy"] = prism_fzz(dz, dx, dy)
            node_evals["gyz"] = prism_fzy(dx, dy, dz)
        if "bz" in components or "tmi" in components:
            if "gxz" not in node_evals:
                node_evals["gxz"] = prism_fzy(dy, dz, dx)
            if "gyz" not in node_evals:
                node_evals["gyz"] = prism_fzy(dx, dy, dz)
            node_evals["gzz"] = prism_fzz(dx, dy, dz)
            # the below will be uncommented when we give the containing cell index
            # for interior observations.
            # if "gxx" not in node_evals or "gyy" not in node_evals:
            #     node_evals["gzz"] = prism_fzz(dx, dy, dz)
            # else:
            #     # This is the one that would need to be adjusted if the observation is
            #     # inside an active cell.
            #     node_evals["gzz"] = -node_evals["gxx"] - node_evals["gyy"]

        if "bxx" in components or "tmi_x" in components:
            node_evals["gxxx"] = prism_fzzz(dy, dz, dx)
            node_evals["gxxy"] = prism_fxxy(dx, dy, dz)
            node_evals["gxxz"] = prism_fxxz(dx, dy, dz)
        if "bxy" in components or "tmi_x" in components or "tmi_y" in components:
            if "gxxy" not in node_evals:
                node_evals["gxxy"] = prism_fxxy(dx, dy, dz)
            node_evals["gyyx"] = prism_fxxz(dy, dz, dx)
            node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
        if "bxz" in components or "tmi_x" in components or "tmi_z" in components:
            if "gxxz" not in node_evals:
                node_evals["gxxz"] = prism_fxxz(dx, dy, dz)
            if "gxyz" not in node_evals:
                node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
            node_evals["gzzx"] = prism_fxxy(dz, dx, dy)
        if "byy" in components or "tmi_y" in components:
            if "gyyx" not in node_evals:
                node_evals["gyyx"] = prism_fxxz(dy, dz, dx)
            node_evals["gyyy"] = prism_fzzz(dz, dx, dy)
            node_evals["gyyz"] = prism_fxxy(dy, dz, dx)
        if "byz" in components or "tmi_y" in components or "tmi_z" in components:
            if "gxyz" not in node_evals:
                node_evals["gxyz"] = prism_fxyz(dx, dy, dz)
            if "gyyz" not in node_evals:
                node_evals["gyyz"] = prism_fxxy(dy, dz, dx)
            node_evals["gzzy"] = prism_fxxz(dz, dx, dy)
        if "bzz" in components or "tmi_z" in components:
            if "gzzx" not in node_evals:
                node_evals["gzzx"] = prism_fxxy(dz, dx, dy)
            if "gzzy" not in node_evals:
                node_evals["gzzy"] = prism_fxxz(dz, dx, dy)
            node_evals["gzzz"] = prism_fzzz(dx, dy, dz)

        ## Hxx = gxxx * m_x + gxxy * m_y + gxxz * m_z
        ## Hxy = gxxy * m_x + gyyx * m_y + gxyz * m_z
        ## Hxz = gxxz * m_x + gxyz * m_y + gzzx * m_z
        ## Hyy = gyyx * m_x + gyyy * m_y + gyyz * m_z
        ## Hyz = gxyz * m_x + gyyz * m_y + gzzy * m_z
        ## Hzz = gzzx * m_x + gzzy * m_y + gzzz * m_z

        rows = {}
        M = self.M
        for component in set(components):
            if component == "bx":
                vals_x = node_evals["gxx"]
                vals_y = node_evals["gxy"]
                vals_z = node_evals["gxz"]
            elif component == "by":
                vals_x = node_evals["gxy"]
                vals_y = node_evals["gyy"]
                vals_z = node_evals["gyz"]
            elif component == "bz":
                vals_x = node_evals["gxz"]
                vals_y = node_evals["gyz"]
                vals_z = node_evals["gzz"]
            elif component == "tmi":
                tmi = self.tmi_projection
                vals_x = (
                    tmi[0] * node_evals["gxx"]
                    + tmi[1] * node_evals["gxy"]
                    + tmi[2] * node_evals["gxz"]
                )
                vals_y = (
                    tmi[0] * node_evals["gxy"]
                    + tmi[1] * node_evals["gyy"]
                    + tmi[2] * node_evals["gyz"]
                )
                vals_z = (
                    tmi[0] * node_evals["gxz"]
                    + tmi[1] * node_evals["gyz"]
                    + tmi[2] * node_evals["gzz"]
                )
            elif component == "tmi_x":
                tmi = self.tmi_projection
                vals_x = (
                    tmi[0] * node_evals["gxxx"]
                    + tmi[1] * node_evals["gxxy"]
                    + tmi[2] * node_evals["gxxz"]
                )
                vals_y = (
                    tmi[0] * node_evals["gxxy"]
                    + tmi[1] * node_evals["gyyx"]
                    + tmi[2] * node_evals["gxyz"]
                )
                vals_z = (
                    tmi[0] * node_evals["gxxz"]
                    + tmi[1] * node_evals["gxyz"]
                    + tmi[2] * node_evals["gzzx"]
                )
            elif component == "tmi_y":
                tmi = self.tmi_projection
                vals_x = (
                    tmi[0] * node_evals["gxxy"]
                    + tmi[1] * node_evals["gyyx"]
                    + tmi[2] * node_evals["gxyz"]
                )
                vals_y = (
                    tmi[0] * node_evals["gyyx"]
                    + tmi[1] * node_evals["gyyy"]
                    + tmi[2] * node_evals["gyyz"]
                )
                vals_z = (
                    tmi[0] * node_evals["gxyz"]
                    + tmi[1] * node_evals["gyyz"]
                    + tmi[2] * node_evals["gzzy"]
                )
            elif component == "tmi_z":
                tmi = self.tmi_projection
                vals_x = (
                    tmi[0] * node_evals["gxxz"]
                    + tmi[1] * node_evals["gxyz"]
                    + tmi[2] * node_evals["gzzx"]
                )
                vals_y = (
                    tmi[0] * node_evals["gxyz"]
                    + tmi[1] * node_evals["gyyz"]
                    + tmi[2] * node_evals["gzzy"]
                )
                vals_z = (
                    tmi[0] * node_evals["gzzx"]
                    + tmi[1] * node_evals["gzzy"]
                    + tmi[2] * node_evals["gzzz"]
                )
            elif component == "bxx":
                vals_x = node_evals["gxxx"]
                vals_y = node_evals["gxxy"]
                vals_z = node_evals["gxxz"]
            elif component == "bxy":
                vals_x = node_evals["gxxy"]
                vals_y = node_evals["gyyx"]
                vals_z = node_evals["gxyz"]
            elif component == "bxz":
                vals_x = node_evals["gxxz"]
                vals_y = node_evals["gxyz"]
                vals_z = node_evals["gzzx"]
            elif component == "byy":
                vals_x = node_evals["gyyx"]
                vals_y = node_evals["gyyy"]
                vals_z = node_evals["gyyz"]
            elif component == "byz":
                vals_x = node_evals["gxyz"]
                vals_y = node_evals["gyyz"]
                vals_z = node_evals["gzzy"]
            elif component == "bzz":
                vals_x = node_evals["gzzx"]
                vals_y = node_evals["gzzy"]
                vals_z = node_evals["gzzz"]
            if self._unique_inv is not None:
                vals_x = vals_x[self._unique_inv]
                vals_y = vals_y[self._unique_inv]
                vals_z = vals_z[self._unique_inv]

            cell_eval_x = (
                vals_x[0]
                - vals_x[1]
                - vals_x[2]
                + vals_x[3]
                - vals_x[4]
                + vals_x[5]
                + vals_x[6]
                - vals_x[7]
            )
            cell_eval_y = (
                vals_y[0]
                - vals_y[1]
                - vals_y[2]
                + vals_y[3]
                - vals_y[4]
                + vals_y[5]
                + vals_y[6]
                - vals_y[7]
            )
            cell_eval_z = (
                vals_z[0]
                - vals_z[1]
                - vals_z[2]
                + vals_z[3]
                - vals_z[4]
                + vals_z[5]
                + vals_z[6]
                - vals_z[7]
            )
            if self.model_type == "vector":
                cell_vals = (
                    np.r_[cell_eval_x, cell_eval_y, cell_eval_z]
                ) * self.survey.source_field.amplitude
            else:
                cell_vals = (
                    cell_eval_x * M[:, 0]
                    + cell_eval_y * M[:, 1]
                    + cell_eval_z * M[:, 2]
                )

            if self.store_sensitivities == "forward_only":
                rows[component] = cell_vals @ self.chi
            else:
                rows[component] = cell_vals

            rows[component] /= 4 * np.pi

        return np.stack(
            [
                rows[component].astype(self.sensitivity_dtype, copy=False)
                for component in components
            ]
        )

    @property
    def deleteTheseOnModelUpdate(self):
        deletes = super().deleteTheseOnModelUpdate
        if self.is_amplitude_data:
            deletes = deletes + ["_gtg_diagonal", "_ampDeriv"]
        return deletes

    def _forward(self, model):
        """
        Forward model the fields of active cells in the mesh on receivers.

        Parameters
        ----------
        model : (n_active_cells) or (3 * n_active_cells) array
            Array containing the susceptibilities (scalar) or effective
            susceptibilities (vector) of the active cells in the mesh, in SI
            units.
            Susceptibilities are expected if ``model_type`` is ``"scalar"``,
            and the array should have ``n_active_cells`` elements.
            Effective susceptibilities are expected if ``model_type`` is
            ``"vector"``, and the array should have ``3 * n_active_cells``
            elements.

        Returns
        -------
        (nD, ) array
            Always return a ``np.float64`` array.
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Allocate fields array
        fields = np.zeros(self.survey.nD, dtype=self.sensitivity_dtype)
        # Define the constant factor
        constant_factor = 1 / 4 / np.pi
        # Start computing the fields
        index_offset = 0
        scalar_model = self.model_type == "scalar"
        for components, receivers in self._get_components_and_receivers():
            if not CHOCLO_SUPPORTED_COMPONENTS.issuperset(components):
                raise NotImplementedError(
                    f"Other components besides {CHOCLO_SUPPORTED_COMPONENTS} "
                    "aren't implemented yet."
                )
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                vector_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                if component == "tmi":
                    self._forward_tmi(
                        receivers,
                        active_nodes,
                        model,
                        fields[vector_slice],
                        active_cell_nodes,
                        regional_field,
                        constant_factor,
                        scalar_model,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    self._forward_tmi_derivative(
                        receivers,
                        active_nodes,
                        model,
                        fields[vector_slice],
                        active_cell_nodes,
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        constant_factor,
                        scalar_model,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    self._forward_mag(
                        receivers,
                        active_nodes,
                        model,
                        fields[vector_slice],
                        active_cell_nodes,
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        constant_factor,
                        scalar_model,
                    )
            index_offset += n_rows
        return fields

    def _sensitivity_matrix(self):
        """
        Compute the sensitivity matrix G

        Returns
        -------
        (nD, n_active_cells) array
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Allocate sensitivity matrix
        if self.model_type == "scalar":
            n_columns = self.nC
        else:
            n_columns = 3 * self.nC
        shape = (self.survey.nD, n_columns)
        if self.store_sensitivities == "disk":
            sensitivity_matrix = np.memmap(
                self.sensitivity_path,
                shape=shape,
                dtype=self.sensitivity_dtype,
                order="C",  # it's more efficient to write in row major
                mode="w+",
            )
        else:
            sensitivity_matrix = np.empty(shape, dtype=self.sensitivity_dtype)
        # Define the constant factor
        constant_factor = 1 / 4 / np.pi
        # Start filling the sensitivity matrix
        index_offset = 0
        scalar_model = self.model_type == "scalar"
        for components, receivers in self._get_components_and_receivers():
            if not CHOCLO_SUPPORTED_COMPONENTS.issuperset(components):
                raise NotImplementedError(
                    f"Other components besides {CHOCLO_SUPPORTED_COMPONENTS} "
                    "aren't implemented yet."
                )
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                matrix_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                if component == "tmi":
                    self._sensitivity_tmi(
                        receivers,
                        active_nodes,
                        sensitivity_matrix[matrix_slice, :],
                        active_cell_nodes,
                        regional_field,
                        constant_factor,
                        scalar_model,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    self._sensitivity_tmi_derivative(
                        receivers,
                        active_nodes,
                        sensitivity_matrix[matrix_slice, :],
                        active_cell_nodes,
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        constant_factor,
                        scalar_model,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    self._sensitivity_mag(
                        receivers,
                        active_nodes,
                        sensitivity_matrix[matrix_slice, :],
                        active_cell_nodes,
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        constant_factor,
                        scalar_model,
                    )
            index_offset += n_rows
        return sensitivity_matrix


class SimulationEquivalentSourceLayer(
    BaseEquivalentSourceLayerSimulation, Simulation3DIntegral
):
    """
    Equivalent source layer simulation

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A 2D tensor or tree mesh defining discretization along the x and y directions
    cell_z_top : numpy.ndarray or float
        Define the elevations for the top face of all cells in the layer.
        If an array it should be the same size as the active cell set.
    cell_z_bottom : numpy.ndarray or float
        Define the elevations for the bottom face of all cells in the layer.
        If an array it should be the same size as the active cell set.
    engine : {"geoana", "choclo"}, optional
        Choose which engine should be used to run the forward model.
    numba_parallel : bool, optional
        If True, the simulation will run in parallel. If False, it will
        run in serial. If ``engine`` is not ``"choclo"`` this argument will be
        ignored.

    """

    def __init__(
        self,
        mesh,
        cell_z_top,
        cell_z_bottom,
        engine="geoana",
        numba_parallel=True,
        **kwargs,
    ):
        super().__init__(
            mesh,
            cell_z_top,
            cell_z_bottom,
            engine=engine,
            numba_parallel=numba_parallel,
            **kwargs,
        )

        if self.engine == "choclo":
            if self.numba_parallel:
                self._sensitivity_tmi = _sensitivity_tmi_2d_mesh_parallel
                self._sensitivity_mag = _sensitivity_mag_2d_mesh_parallel
                self._forward_tmi = _forward_tmi_2d_mesh_parallel
                self._forward_mag = _forward_mag_2d_mesh_parallel
                self._forward_tmi_derivative = _forward_tmi_derivative_2d_mesh_parallel
                self._sensitivity_tmi_derivative = (
                    _sensitivity_tmi_derivative_2d_mesh_parallel
                )
            else:
                self._sensitivity_tmi = _sensitivity_tmi_2d_mesh_serial
                self._sensitivity_mag = _sensitivity_mag_2d_mesh_serial
                self._forward_tmi = _forward_tmi_2d_mesh_serial
                self._forward_mag = _forward_mag_2d_mesh_serial
                self._forward_tmi_derivative = _forward_tmi_derivative_2d_mesh_serial
                self._sensitivity_tmi_derivative = (
                    _sensitivity_tmi_derivative_2d_mesh_serial
                )

    def _forward(self, model):
        """
        Forward model the fields of active cells in the mesh on receivers.

        Parameters
        ----------
        model : (n_active_cells) or (3 * n_active_cells) array
            Array containing the susceptibilities (scalar) or effective
            susceptibilities (vector) of the active cells in the mesh, in SI
            units.
            Susceptibilities are expected if ``model_type`` is ``"scalar"``,
            and the array should have ``n_active_cells`` elements.
            Effective susceptibilities are expected if ``model_type`` is
            ``"vector"``, and the array should have ``3 * n_active_cells``
            elements.

        Returns
        -------
        (nD, ) array
            Always return a ``np.float64`` array.
        """
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Allocate fields array
        fields = np.zeros(self.survey.nD, dtype=self.sensitivity_dtype)
        # Start computing the fields
        index_offset = 0
        scalar_model = self.model_type == "scalar"
        for components, receivers in self._get_components_and_receivers():
            if not CHOCLO_SUPPORTED_COMPONENTS.issuperset(components):
                raise NotImplementedError(
                    f"Other components besides {CHOCLO_SUPPORTED_COMPONENTS} "
                    "aren't implemented yet."
                )
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                vector_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                if component == "tmi":
                    self._forward_tmi(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        model,
                        fields[vector_slice],
                        regional_field,
                        scalar_model,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    self._forward_tmi_derivative(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        model,
                        fields[vector_slice],
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        scalar_model,
                    )
                else:
                    forward_func = CHOCLO_FORWARD_FUNCS[component]
                    self._forward_mag(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        model,
                        fields[vector_slice],
                        regional_field,
                        forward_func,
                        scalar_model,
                    )
            index_offset += n_rows
        return fields

    def _sensitivity_matrix(self):
        """
        Compute the sensitivity matrix G

        Returns
        -------
        (nD, n_active_cells) array
        """
        # Get cells in the 2D mesh and keep only active cells
        cells_bounds_active = self.mesh.cell_bounds[self.active_cells]
        # Get regional field
        regional_field = self.survey.source_field.b0
        # Allocate sensitivity matrix
        if self.model_type == "scalar":
            n_columns = self.nC
        else:
            n_columns = 3 * self.nC
        shape = (self.survey.nD, n_columns)
        if self.store_sensitivities == "disk":
            sensitivity_matrix = np.memmap(
                self.sensitivity_path,
                shape=shape,
                dtype=self.sensitivity_dtype,
                order="C",  # it's more efficient to write in row major
                mode="w+",
            )
        else:
            sensitivity_matrix = np.empty(shape, dtype=self.sensitivity_dtype)
        # Start filling the sensitivity matrix
        index_offset = 0
        scalar_model = self.model_type == "scalar"
        for components, receivers in self._get_components_and_receivers():
            if not CHOCLO_SUPPORTED_COMPONENTS.issuperset(components):
                raise NotImplementedError(
                    f"Other components besides {CHOCLO_SUPPORTED_COMPONENTS} "
                    "aren't implemented yet."
                )
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                matrix_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                if component == "tmi":
                    self._sensitivity_tmi(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        sensitivity_matrix[matrix_slice, :],
                        regional_field,
                        scalar_model,
                    )
                elif component in ("tmi_x", "tmi_y", "tmi_z"):
                    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz = (
                        CHOCLO_KERNELS[component]
                    )
                    self._sensitivity_tmi_derivative(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        sensitivity_matrix[matrix_slice, :],
                        regional_field,
                        kernel_xx,
                        kernel_yy,
                        kernel_zz,
                        kernel_xy,
                        kernel_xz,
                        kernel_yz,
                        scalar_model,
                    )
                else:
                    kernel_x, kernel_y, kernel_z = CHOCLO_KERNELS[component]
                    self._sensitivity_mag(
                        receivers,
                        cells_bounds_active,
                        self.cell_z_top,
                        self.cell_z_bottom,
                        sensitivity_matrix[matrix_slice, :],
                        regional_field,
                        kernel_x,
                        kernel_y,
                        kernel_z,
                        scalar_model,
                    )
            index_offset += n_rows
        return sensitivity_matrix

class Simulation3DDifferential(BaseMagneticPDESimulation):
    r"""A secondary field simulation for magnetic data.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    survey : magnetics.suvey.Survey
    mu : float, array_like
        Magnetic Permeability Model (C/(s m^3)). Set this for forward
        modeling or to fix while inverting for remanence. This is used if
        muMap == None
    rem : float, array_like
        Magnetic Polarization \mu_0*M (nT). Set this for forward
        modeling or to fix remanent magnetization while inverting for permeability.
        This is used if remMap == None
    muMap : SimPEG.maps.IdentityMap, optional
        The mapping used to go from the simulation model to `mu`. Set this
        to invert for `mu`.
    remMap : SimPEG.maps.IdentityMap, optional
        The mapping used to go from the simulation model to `mu_0*M`. Set this
        to invert for `mu_0*M`.
    storeJ: bool
        Whether to store the sensitivity matrix. If set to True


    Notes
    -----
    This simulation solves for the magnetostatic PDE:
    \nabla \cdot \Vec{B} = 0

    where the constitutive relation is defined as:
    \Vec{B} = \mu\Vec{H} + \mu_0\Vec{M_r}

    where \Vec{M_r} is a fixed magnetization unaffected by the inducing field
    and \mu\Vec{H} is the induced magnetization
    """

    _Jmatrix = None
    _Ainv = None
    _stored_fields = None

    rem, remMap, remDeriv = props.Invertible(
        "Magnetic Polarization (nT)", optional=True
    )

    def __init__(
        self,
        mesh,
        survey=None,
        mu=None,
        rem=None,
        remMap=None,
        muMap=None,
        storeJ=False,
        **kwargs
    ):
        if mu is None:
            mu = mu_0

        super().__init__(mesh=mesh, survey=survey, mu=mu, muMap=muMap, **kwargs)

        if (
            muMap is None
            and np.isscalar(mu)
            and np.allclose(mu, mu_0)
            and storeJ is True
        ):
            self._update_J = False
        else:
            self._update_J = True

        self.rem = rem
        self.remMap = remMap

        self.storeJ = storeJ

        self._MfMu0i = self.mesh.get_face_inner_product(1.0 / mu_0)
        self._Div = self.Mcc * self.mesh.face_divergence
        self._DivT = self._Div.T.tocsr()
        self._Mf_vec_deriv = self.mesh.get_face_inner_product_deriv(
            np.ones(self.mesh.n_cells * 3)
        )(np.ones(self.mesh.n_faces))

        self.solver_opts = {"is_symmetric": True, "is_positive_definite": True}

    @property
    def survey(self):
        """The magnetic survey object.

        Returns
        -------
        SimPEG.potential_fields.magnetics.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        self._survey = value

    @property
    def storeJ(self):
        return self._storeJ

    @storeJ.setter
    def storeJ(self, value):
        self._storeJ = validate_type("storeJ", value, bool)

    @property
    def exact_TMI(self):
        return self._exact_TMI

    @exact_TMI.setter
    def exact_TMI(self, value):
        self._exact_TMI = validate_type("exact_TMI", value, bool)

    @utils.requires("survey")
    def getB0(self):
        b0 = self.survey.source_field.b0
        B0 = np.r_[
            b0[0] * np.ones(self.mesh.nFx),
            b0[1] * np.ones(self.mesh.nFy),
            b0[2] * np.ones(self.mesh.nFz),
        ]
        return B0

    def getRHS(self, m):
        self.model = m

        rhs = 0

        if not np.isscalar(self.mu) or not np.allclose(self.mu, mu_0):
            B0 = self.getB0()
            rhs += self._Div * self.MfMuiI * self._MfMu0i * B0 - self._Div * B0

        if self.rem is not None:
            mu = self.mu * np.ones(self.mesh.n_cells)
            mu_vec = np.hstack((mu, mu, mu))
            rhs += (
                self._Div
                * (
                    self.MfMuiI * self.mesh.get_face_inner_product(self.rem / mu_vec)
                ).diagonal()
            )

        return rhs

    def getA(self, m):
        return self._Div * self.MfMuiI * self._DivT

    def fields(self, m):
        self.model = m

        if self._stored_fields is None:

            if self._Ainv is None:
                self._Ainv = self.solver(self.getA(m), **self.solver_opts)

            rhs = self.getRHS(m)

            u = self._Ainv * rhs
            B = -self.MfMuiI * self._DivT * u

            if not np.isscalar(self.mu) or not np.allclose(self.mu, mu_0):
                B0 = self.getB0()
                B += self._MfMu0i * self.MfMuiI * B0 - B0

            if self.rem is not None:
                mu = self.mu * np.ones(self.mesh.n_cells)
                mu_vec = np.hstack((mu, mu, mu))
                B += (
                    self.MfMuiI * self.mesh.get_face_inner_product(self.rem / mu_vec)
                ).diagonal()

            fields = {"B": B, "u": u}
            self._stored_fields = fields

        else:
            fields = self._stored_fields

        return fields

    def dpred(self, m=None, f=None):
        if f is not None:
            return self.projectFields(f)

        if f is None:
            f = self.fields(m)

        dpred = self.projectFields(f)

        return dpred
    def Jvec(self, m, v, f=None):
        self.model = m

        if self.storeJ:
            J = self.getJ(m, f=f)
            return J.dot(v)

        self.model = m

        if f is None:
            f = self.fields(m)

        return self._Jvec(m, v, f)

    def Jtvec(self, m, v, f=None):
        self.model = m

        if self.storeJ:
            J = self.getJ(m, f=f)
            return np.asarray(J.T.dot(v))

        self.model = m

        if f is None:
            f = self.fields(m)

        return self._Jtvec(m, v, f)

    def getJ(self, m, f=None):
        if self._Jmatrix is None:
            if f is None:
                f = self.fields(m)
            if m.size < self.survey.nD:
                J = self._Jvec(m, v=None, f=f)
            else:
                J = self._Jtvec(m, v=None, f=f).T

        else:
            J = self._Jmatrix

        if self.storeJ is True:
            self._Jmatrix = J

        return J

    def _Jtvec(self, m, v, f=None):
        B, u = f["B"], f["u"]

        Q = self.projectFieldsDeriv(B)

        B0 = self.getB0()
        if v is None:
            v = np.eye(Q.shape[0])
            DivTatsol_p_QT = (
                self._DivT * (self._Ainv * ((Q * self.MfMuiI * -self._DivT).T * v))
                + Q.T * v
            )
        else:
            DivTatsol_p_QT = (
                self._DivT * (self._Ainv * ((-self._Div * (self.MfMuiI.T * (Q.T * v)))))
                + Q.T * v
            )

        del Q

        mu = np.ones(self.mesh.n_cells) * self.mu
        mu_vec = np.hstack((mu, mu, mu))

        Jtv = 0

        if self.remMap is not None:
            Mf_rem_deriv = self._Mf_vec_deriv * sp.diags(1 / mu_vec) * self.remDeriv
            Jtv += (self.MfMuiI * Mf_rem_deriv).T * (DivTatsol_p_QT)

        if self.muMap is not None:
            Jtv += self.MfMuiIDeriv(self._DivT * u, -DivTatsol_p_QT, adjoint=True)
            Jtv += self.MfMuiIDeriv(B0, self._MfMu0i.T * (DivTatsol_p_QT), adjoint=True)

            if self.rem is not None:
                Mf_r_over_uvec = self.mesh.get_face_inner_product(
                    self.rem / mu_vec
                ).diagonal()
                mu_vec_i_deriv = sp.vstack(
                    (self.muiDeriv, self.muiDeriv, self.muiDeriv)
                )

                Mf_mu_vec_i_deriv = (
                    self._Mf_vec_deriv * sp.diags(self.rem) * mu_vec_i_deriv
                )

                Jtv += (
                    self.MfMuiIDeriv(Mf_r_over_uvec, DivTatsol_p_QT, adjoint=True)
                    + (Mf_mu_vec_i_deriv.T * self.MfMuiI.T) * DivTatsol_p_QT
                )

        return Jtv


    def _Jvec(self, m, v, f=None):

        if v is None:
            v = np.eye(m.shape[0])

        B, u = f["B"], f["u"]

        Q = self.projectFieldsDeriv(B)
        B0 = self.getB0()
        C = -self.MfMuiI * self._DivT

        db_dm = 0
        dCmu_dm = 0

        mu = np.ones(self.mesh.n_cells) * self.mu
        mu_vec = np.hstack((mu, mu, mu))

        if self.remMap is not None:
            Mf_rem_deriv = self._Mf_vec_deriv * sp.diags(1 / mu_vec) * self.remDeriv
            db_dm += self.MfMuiI * Mf_rem_deriv * v

        if self.muMap is not None:
            dCmu_dm += self.MfMuiIDeriv(self._DivT @ u, v, adjoint=False)
            db_dm += self._MfMu0i * self.MfMuiIDeriv(B0, v, adjoint=False)

            if self.rem is not None:
                Mf_r_over_uvec = self.mesh.get_face_inner_product(
                    self.rem / mu_vec
                ).diagonal()
                mu_vec_i_deriv = sp.vstack(
                    (self.muiDeriv, self.muiDeriv, self.muiDeriv)
                )
                Mf_mu_vec_i_deriv = (
                    self._Mf_vec_deriv * sp.diags(self.rem) * mu_vec_i_deriv
                )
                db_dm += self.MfMuiIDeriv(Mf_r_over_uvec, v, adjoint=False) + (
                    self.MfMuiI * Mf_mu_vec_i_deriv * v
                )

        dq_dm_min_dAmu_dm = self._Div * (-dCmu_dm + db_dm)

        Ainv_dqmindAmu = self._Ainv * dq_dm_min_dAmu_dm

        Jv = Q * (C * Ainv_dqmindAmu + (-dCmu_dm + db_dm))

        return Jv

    @property
    def Qfx(self):
        if getattr(self, "_Qfx", None) is None:
            self._Qfx = self.mesh.get_interpolation_matrix(
                self.survey.receiver_locations, "Fx"
            )
        return self._Qfx

    @property
    def Qfy(self):
        if getattr(self, "_Qfy", None) is None:
            self._Qfy = self.mesh.get_interpolation_matrix(
                self.survey.receiver_locations, "Fy"
            )
        return self._Qfy

    @property
    def Qfz(self):
        if getattr(self, "_Qfz", None) is None:
            self._Qfz = self.mesh.get_interpolation_matrix(
                self.survey.receiver_locations, "Fz"
            )
        return self._Qfz

    def projectFields(self, u):
        components = self.survey.components

        fields = {}


        if "bx" in components or "tmi" in components:
            fields["bx"] = self.Qfx * u["B"]
        if "by" in components or "tmi" in components:
            fields["by"] = self.Qfy * u["B"]
        if "bz" in components or "tmi" in components:
            fields["bz"] = self.Qfz * u["B"]

        B0 = self.survey.source_field.b0

        if "tmi" in components:

            Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)
            box = B0[0]
            boy = B0[1]
            boz = B0[2]
            fields["tmi"] = (
                np.sqrt(
                    (fields["bx"] + box) ** 2
                    + (fields["by"] + boy) ** 2
                    + (fields["bz"] + boz) ** 2
                )
                - Bot
            )


        return np.concatenate([fields[comp] for comp in components])

    @utils.count
    def projectFieldsDeriv(self, Bs):
        components = self.survey.components

        fields = {}

        if "bx" in components or "tmi" in components:
            fields["bx"] = self.Qfx
        if "by" in components or "tmi" in components:
            fields["by"] = self.Qfy
        if "bz" in components or "tmi" in components:
            fields["bz"] = self.Qfz

        if "tmi" in components:
            B0 = self.survey.source_field.b0
            Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)

            if self.exact_TMI:
                if not self._update_J and self._Jmatrix is None:
                    return sp.vstack((self.Qfx, self.Qfy, self.Qfz))

                box = B0[0]
                boy = B0[1]
                boz = B0[2]

                bx = self.Qfx * Bs
                by = self.Qfy * Bs
                bz = self.Qfz * Bs

                dpred = (
                    np.sqrt((bx + box) ** 2 + (by + boy) ** 2 + (bz + boz) ** 2) - Bot
                )

                dDhalf_dD = sdiag(1 / (dpred + Bot))

                xterm = sdiag(box + bx) * self.Qfx
                yterm = sdiag(boy + by) * self.Qfy
                zterm = sdiag(boz + bz) * self.Qfz

                fields["tmi"] = dDhalf_dD * (xterm + yterm + zterm)

            else:
                bx = fields["bx"]
                by = fields["by"]
                bz = fields["bz"]
                B0 = self.survey.source_field.b0
                Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)
                box = B0[0] / Bot
                boy = B0[1] / Bot
                boz = B0[2] / Bot
                fields["tmi"] = bx * box + by * boy + bz * boz

        return sp.vstack([fields[comp] for comp in components])

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super().deleteTheseOnModelUpdate
        if self._stored_fields is not None:
            toDelete = toDelete + ["_stored_fields"]
        if self._Ainv is not None:
            toDelete = toDelete + ["_Ainv"]
        if self._update_J:
            if self._Jmatrix is not None:
                toDelete = toDelete + ["_Jmatrix"]
        return toDelete

    @property
    def clean_on_model_update(self):
        toclean = super().clean_on_model_update
        if self.muMap is None:
            return toclean
        else:
            return toclean + ["_Ainv"]

