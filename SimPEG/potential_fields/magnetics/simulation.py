import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
import numpy_indexed as npi

from SimPEG import utils
from ..base import BasePFSimulation
from ...base import BaseMagneticPDESimulation
from .survey import Survey
from .analytics import CongruousMagBC

from SimPEG import Solver
from SimPEG import props
import properties
from SimPEG.utils import mkvc, mat_utils, sdiag, setKwargs


class Simulation3DIntegral(BasePFSimulation):
    """
    magnetic simulation in integral form.

    """

    chi, chiMap, chiDeriv = props.Invertible(
        "Magnetic Susceptibility (SI)", default=1.0
    )

    is_amplitude_data = properties.Boolean(
        "Whether the supplied data is amplitude data", default=False
    )

    _model_type: str = "scalar"

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self._G = None
        self._M = None
        self._gtg_diagonal = None
        self.modelMap = self.chiMap
        setKwargs(self, **kwargs)

    @property
    def M(self):
        """
        M: ndarray
            Magnetization matrix
        """
        if getattr(self, "_M", None) is None:

            if self.model_type == "vector":
                self._M = sp.identity(self.nC) * self.survey.source_field.parameters[0]

            else:
                mag = mat_utils.dip_azimuth2cartesian(
                    np.ones(self.nC) * self.survey.source_field.parameters[1],
                    np.ones(self.nC) * self.survey.source_field.parameters[2],
                )

                self._M = sp.vstack(
                    (
                        sdiag(mag[:, 0] * self.survey.source_field.parameters[0]),
                        sdiag(mag[:, 1] * self.survey.source_field.parameters[0]),
                        sdiag(mag[:, 2] * self.survey.source_field.parameters[0]),
                    )
                )

        return self._M

    @M.setter
    def M(self, M):
        """
        Create magnetization matrix from unit vector orientation
        :parameter
        M: array (3*nC,) or (nC, 3)
        """
        if self.model_type == "vector":
            self._M = sdiag(mkvc(M) * self.survey.source_field.parameters[0])
        else:
            M = M.reshape((-1, 3))
            self._M = sp.vstack(
                (
                    sdiag(M[:, 0] * self.survey.source_field.parameters[0]),
                    sdiag(M[:, 1] * self.survey.source_field.parameters[0]),
                    sdiag(M[:, 2] * self.survey.source_field.parameters[0]),
                )
            )

    def fields(self, model):

        model = self.chiMap * model

        if self.store_sensitivities == "forward_only":
            self.model = model
            fields = mkvc(self.linear_operator())
        else:
            fields = np.asarray(self.G @ model.astype(np.float32))

        if self.is_amplitude_data:
            fields = self.compute_amplitude(fields)

        return fields

    @property
    def G(self):

        if getattr(self, "_G", None) is None:

            self._G = self.linear_operator()

        return self._G

    @property
    def model_type(self) -> str:
        """
        Define the type of model. Choice of 'scalar' or 'vector' (3-components)
        """
        return self._model_type

    @model_type.setter
    def model_type(self, value: str):
        if value not in ["scalar", "vector"]:
            raise ValueError(
                "'model_type' value should be a string: 'scalar' or 'vector'."
                + f"Value {value} of type {type(value)} provided."
            )

        self._model_type = value

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
                self.survey.source_field.parameters[1],
                self.survey.source_field.parameters[2],
            )

        return self._tmi_projection

    def getJtJdiag(self, m, W=None):
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
                fieldDeriv = self.fieldDeriv
                Gx = self.G[::3]
                Gy = self.G[1::3]
                Gz = self.G[2::3]
                for i in range(len(W)):
                    row = (
                        fieldDeriv[0, i] * Gx[i]
                        + fieldDeriv[1, i] * Gy[i]
                        + fieldDeriv[2, i] * Gz[i]
                    )
                    diag += W[i] * (row * row)
            self._gtg_diagonal = diag
        else:
            diag = self._gtg_diagonal
        return mkvc((sdiag(np.sqrt(diag)) @ self.chiDeriv).power(2).sum(axis=0))

    def Jvec(self, m, v, f=None):
        self.model = m
        dmu_dm_v = self.chiDeriv @ v

        Jvec = self.G @ dmu_dm_v.astype(np.float32)

        if self.is_amplitude_data:
            Jvec = Jvec.reshape((-1, 3)).T
            fieldDeriv_Jvec = self.fieldDeriv * Jvec
            return fieldDeriv_Jvec[0] + fieldDeriv_Jvec[1] + fieldDeriv_Jvec[2]
        else:
            return Jvec

    def Jtvec(self, m, v, f=None):
        self.model = m

        if self.is_amplitude_data:
            v = (self.fieldDeriv * v).T.reshape(-1)
        Jtvec = self.G.T @ v.astype(np.float32)
        return np.asarray(self.chiDeriv.T @ Jtvec)

    @property
    def fieldDeriv(self):

        if getattr(self, "chi", None) is None:
            self.model = np.zeros(self.chiMap.nP)

        if getattr(self, "_fieldDeriv", None) is None:
            fields = np.asarray(self.G.dot((self.chiMap @ self.chi).astype(np.float32)))
            b_xyz = self.normalized_fields(fields)

            self._fieldDeriv = b_xyz

        return self._fieldDeriv

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

    def evaluate_integral(self, receiver_location, components, tolerance=1e-4):
        """
        Load in the active nodes of a tensor mesh and computes the magnetic
        forward relation between a cuboid and a given observation
        location outside the Earth [obsx, obsy, obsz]

        INPUT:
        receiver_location:  [obsx, obsy, obsz] nC x 3 Array

        components: list[str]
            List of magnetic components chosen from:
            'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz'

        tolerance: float
            Small constant to avoid singularity near nodes and edges.

        OUTPUT:
        Tx = [Txx Txy Txz]
        Ty = [Tyx Tyy Tyz]
        Tz = [Tzx Tzy Tzz]
        """
        # TODO: This should probably be converted to C
        rows = {component: np.zeros(3 * self.Xn.shape[0]) for component in components}

        # number of cells in mesh
        nC = self.Xn.shape[0]

        # base cell dimensions
        min_hx, min_hy = self.mesh.h[0].min(), self.mesh.h[1].min()
        if len(self.mesh.h) < 3:
            # Allow for 2D quadtree representations by using a dummy cell height.
            # Actually cell heights will come from externally defined ``self.Zn``
            min_hz = np.minimum(min_hx, min_hy) / 10.0
        else:
            min_hz = self.mesh.h[2].min()

        # comp. pos. differences for tne, bsw nodes. Adjust if location within
        # tolerance of a node or edge
        dx1 = self.Xn[:, 0] - receiver_location[0]
        dx1[np.abs(dx1) / min_hx < tolerance] = tolerance * min_hx
        dx2 = self.Xn[:, 1] - receiver_location[0]
        dx2[np.abs(dx2) / min_hx < tolerance] = tolerance * min_hx

        dy1 = self.Yn[:, 0] - receiver_location[1]
        dy1[np.abs(dy1) / min_hy < tolerance] = tolerance * min_hy
        dy2 = self.Yn[:, 1] - receiver_location[1]
        dy2[np.abs(dy2) / min_hy < tolerance] = tolerance * min_hy

        dz1 = self.Zn[:, 0] - receiver_location[2]
        dz1[np.abs(dz1) / min_hz < tolerance] = tolerance * min_hz
        dz2 = self.Zn[:, 1] - receiver_location[2]
        dz2[np.abs(dz2) / min_hz < tolerance] = tolerance * min_hz

        # comp. squared diff
        dx2dx2 = dx2**2.0
        dx1dx1 = dx1**2.0

        dy2dy2 = dy2**2.0
        dy1dy1 = dy1**2.0

        dz2dz2 = dz2**2.0
        dz1dz1 = dz1**2.0

        # 2D radius component squared of corner nodes
        R1 = dy2dy2 + dx2dx2
        R2 = dy2dy2 + dx1dx1
        R3 = dy1dy1 + dx2dx2
        R4 = dy1dy1 + dx1dx1

        # radius to each cell node
        r1 = np.sqrt(dz2dz2 + R2)
        r2 = np.sqrt(dz2dz2 + R1)
        r3 = np.sqrt(dz1dz1 + R1)
        r4 = np.sqrt(dz1dz1 + R2)
        r5 = np.sqrt(dz2dz2 + R3)
        r6 = np.sqrt(dz2dz2 + R4)
        r7 = np.sqrt(dz1dz1 + R4)
        r8 = np.sqrt(dz1dz1 + R3)

        # compactify argument calculations
        arg1_ = dx1 + dy2 + r1
        arg1 = dy2 + dz2 + r1
        arg2 = dx1 + dz2 + r1
        arg3 = dx1 + r1
        arg4 = dy2 + r1
        arg5 = dz2 + r1

        arg6_ = dx2 + dy2 + r2
        arg6 = dy2 + dz2 + r2
        arg7 = dx2 + dz2 + r2
        arg8 = dx2 + r2
        arg9 = dy2 + r2
        arg10 = dz2 + r2

        arg11_ = dx2 + dy2 + r3
        arg11 = dy2 + dz1 + r3
        arg12 = dx2 + dz1 + r3
        arg13 = dx2 + r3
        arg14 = dy2 + r3
        arg15 = dz1 + r3

        arg16_ = dx1 + dy2 + r4
        arg16 = dy2 + dz1 + r4
        arg17 = dx1 + dz1 + r4
        arg18 = dx1 + r4
        arg19 = dy2 + r4
        arg20 = dz1 + r4

        arg21_ = dx2 + dy1 + r5
        arg21 = dy1 + dz2 + r5
        arg22 = dx2 + dz2 + r5
        arg23 = dx2 + r5
        arg24 = dy1 + r5
        arg25 = dz2 + r5

        arg26_ = dx1 + dy1 + r6
        arg26 = dy1 + dz2 + r6
        arg27 = dx1 + dz2 + r6
        arg28 = dx1 + r6
        arg29 = dy1 + r6
        arg30 = dz2 + r6

        arg31_ = dx1 + dy1 + r7
        arg31 = dy1 + dz1 + r7
        arg32 = dx1 + dz1 + r7
        arg33 = dx1 + r7
        arg34 = dy1 + r7
        arg35 = dz1 + r7

        arg36_ = dx2 + dy1 + r8
        arg36 = dy1 + dz1 + r8
        arg37 = dx2 + dz1 + r8
        arg38 = dx2 + r8
        arg39 = dy1 + r8
        arg40 = dz1 + r8

        if ("bxx" in components) or ("bzz" in components):
            rows["bxx"] = np.zeros((1, 3 * nC))

            rows["bxx"][0, 0:nC] = 2 * (
                ((dx1**2 - r1 * arg1) / (r1 * arg1**2 + dx1**2 * r1))
                - ((dx2**2 - r2 * arg6) / (r2 * arg6**2 + dx2**2 * r2))
                + ((dx2**2 - r3 * arg11) / (r3 * arg11**2 + dx2**2 * r3))
                - ((dx1**2 - r4 * arg16) / (r4 * arg16**2 + dx1**2 * r4))
                + ((dx2**2 - r5 * arg21) / (r5 * arg21**2 + dx2**2 * r5))
                - ((dx1**2 - r6 * arg26) / (r6 * arg26**2 + dx1**2 * r6))
                + ((dx1**2 - r7 * arg31) / (r7 * arg31**2 + dx1**2 * r7))
                - ((dx2**2 - r8 * arg36) / (r8 * arg36**2 + dx2**2 * r8))
            )

            rows["bxx"][0, nC : 2 * nC] = (
                dx2 / (r5 * arg25)
                - dx2 / (r2 * arg10)
                + dx2 / (r3 * arg15)
                - dx2 / (r8 * arg40)
                + dx1 / (r1 * arg5)
                - dx1 / (r6 * arg30)
                + dx1 / (r7 * arg35)
                - dx1 / (r4 * arg20)
            )

            rows["bxx"][0, 2 * nC :] = (
                dx1 / (r1 * arg4)
                - dx2 / (r2 * arg9)
                + dx2 / (r3 * arg14)
                - dx1 / (r4 * arg19)
                + dx2 / (r5 * arg24)
                - dx1 / (r6 * arg29)
                + dx1 / (r7 * arg34)
                - dx2 / (r8 * arg39)
            )

            rows["bxx"] /= 4 * np.pi
            rows["bxx"] *= self.M

        if ("byy" in components) or ("bzz" in components):

            rows["byy"] = np.zeros((1, 3 * nC))

            rows["byy"][0, 0:nC] = (
                dy2 / (r3 * arg15)
                - dy2 / (r2 * arg10)
                + dy1 / (r5 * arg25)
                - dy1 / (r8 * arg40)
                + dy2 / (r1 * arg5)
                - dy2 / (r4 * arg20)
                + dy1 / (r7 * arg35)
                - dy1 / (r6 * arg30)
            )
            rows["byy"][0, nC : 2 * nC] = 2 * (
                ((dy2**2 - r1 * arg2) / (r1 * arg2**2 + dy2**2 * r1))
                - ((dy2**2 - r2 * arg7) / (r2 * arg7**2 + dy2**2 * r2))
                + ((dy2**2 - r3 * arg12) / (r3 * arg12**2 + dy2**2 * r3))
                - ((dy2**2 - r4 * arg17) / (r4 * arg17**2 + dy2**2 * r4))
                + ((dy1**2 - r5 * arg22) / (r5 * arg22**2 + dy1**2 * r5))
                - ((dy1**2 - r6 * arg27) / (r6 * arg27**2 + dy1**2 * r6))
                + ((dy1**2 - r7 * arg32) / (r7 * arg32**2 + dy1**2 * r7))
                - ((dy1**2 - r8 * arg37) / (r8 * arg37**2 + dy1**2 * r8))
            )
            rows["byy"][0, 2 * nC :] = (
                dy2 / (r1 * arg3)
                - dy2 / (r2 * arg8)
                + dy2 / (r3 * arg13)
                - dy2 / (r4 * arg18)
                + dy1 / (r5 * arg23)
                - dy1 / (r6 * arg28)
                + dy1 / (r7 * arg33)
                - dy1 / (r8 * arg38)
            )

            rows["byy"] /= 4 * np.pi
            rows["byy"] *= self.M

        if "bzz" in components:

            rows["bzz"] = -rows["bxx"] - rows["byy"]

        if "bxy" in components:
            rows["bxy"] = np.zeros((1, 3 * nC))

            rows["bxy"][0, 0:nC] = 2 * (
                ((dx1 * arg4) / (r1 * arg1**2 + (dx1**2) * r1))
                - ((dx2 * arg9) / (r2 * arg6**2 + (dx2**2) * r2))
                + ((dx2 * arg14) / (r3 * arg11**2 + (dx2**2) * r3))
                - ((dx1 * arg19) / (r4 * arg16**2 + (dx1**2) * r4))
                + ((dx2 * arg24) / (r5 * arg21**2 + (dx2**2) * r5))
                - ((dx1 * arg29) / (r6 * arg26**2 + (dx1**2) * r6))
                + ((dx1 * arg34) / (r7 * arg31**2 + (dx1**2) * r7))
                - ((dx2 * arg39) / (r8 * arg36**2 + (dx2**2) * r8))
            )
            rows["bxy"][0, nC : 2 * nC] = (
                dy2 / (r1 * arg5)
                - dy2 / (r2 * arg10)
                + dy2 / (r3 * arg15)
                - dy2 / (r4 * arg20)
                + dy1 / (r5 * arg25)
                - dy1 / (r6 * arg30)
                + dy1 / (r7 * arg35)
                - dy1 / (r8 * arg40)
            )
            rows["bxy"][0, 2 * nC :] = (
                1 / r1 - 1 / r2 + 1 / r3 - 1 / r4 + 1 / r5 - 1 / r6 + 1 / r7 - 1 / r8
            )

            rows["bxy"] /= 4 * np.pi

            rows["bxy"] *= self.M

        if "bxz" in components:
            rows["bxz"] = np.zeros((1, 3 * nC))

            rows["bxz"][0, 0:nC] = 2 * (
                ((dx1 * arg5) / (r1 * (arg1**2) + (dx1**2) * r1))
                - ((dx2 * arg10) / (r2 * (arg6**2) + (dx2**2) * r2))
                + ((dx2 * arg15) / (r3 * (arg11**2) + (dx2**2) * r3))
                - ((dx1 * arg20) / (r4 * (arg16**2) + (dx1**2) * r4))
                + ((dx2 * arg25) / (r5 * (arg21**2) + (dx2**2) * r5))
                - ((dx1 * arg30) / (r6 * (arg26**2) + (dx1**2) * r6))
                + ((dx1 * arg35) / (r7 * (arg31**2) + (dx1**2) * r7))
                - ((dx2 * arg40) / (r8 * (arg36**2) + (dx2**2) * r8))
            )
            rows["bxz"][0, nC : 2 * nC] = (
                1 / r1 - 1 / r2 + 1 / r3 - 1 / r4 + 1 / r5 - 1 / r6 + 1 / r7 - 1 / r8
            )
            rows["bxz"][0, 2 * nC :] = (
                dz2 / (r1 * arg4)
                - dz2 / (r2 * arg9)
                + dz1 / (r3 * arg14)
                - dz1 / (r4 * arg19)
                + dz2 / (r5 * arg24)
                - dz2 / (r6 * arg29)
                + dz1 / (r7 * arg34)
                - dz1 / (r8 * arg39)
            )

            rows["bxz"] /= 4 * np.pi

            rows["bxz"] *= self.M

        if "byz" in components:
            rows["byz"] = np.zeros((1, 3 * nC))

            rows["byz"][0, 0:nC] = (
                1 / r3 - 1 / r2 + 1 / r5 - 1 / r8 + 1 / r1 - 1 / r4 + 1 / r7 - 1 / r6
            )
            rows["byz"][0, nC : 2 * nC] = 2 * (
                (((dy2 * arg5) / (r1 * (arg2**2) + (dy2**2) * r1)))
                - (((dy2 * arg10) / (r2 * (arg7**2) + (dy2**2) * r2)))
                + (((dy2 * arg15) / (r3 * (arg12**2) + (dy2**2) * r3)))
                - (((dy2 * arg20) / (r4 * (arg17**2) + (dy2**2) * r4)))
                + (((dy1 * arg25) / (r5 * (arg22**2) + (dy1**2) * r5)))
                - (((dy1 * arg30) / (r6 * (arg27**2) + (dy1**2) * r6)))
                + (((dy1 * arg35) / (r7 * (arg32**2) + (dy1**2) * r7)))
                - (((dy1 * arg40) / (r8 * (arg37**2) + (dy1**2) * r8)))
            )
            rows["byz"][0, 2 * nC :] = (
                dz2 / (r1 * arg3)
                - dz2 / (r2 * arg8)
                + dz1 / (r3 * arg13)
                - dz1 / (r4 * arg18)
                + dz2 / (r5 * arg23)
                - dz2 / (r6 * arg28)
                + dz1 / (r7 * arg33)
                - dz1 / (r8 * arg38)
            )

            rows["byz"] /= 4 * np.pi

            rows["byz"] *= self.M

        if ("bx" in components) or ("tmi" in components):
            rows["bx"] = np.zeros((1, 3 * nC))

            rows["bx"][0, 0:nC] = (
                (-2 * np.arctan2(dx1, arg1 + tolerance))
                - (-2 * np.arctan2(dx2, arg6 + tolerance))
                + (-2 * np.arctan2(dx2, arg11 + tolerance))
                - (-2 * np.arctan2(dx1, arg16 + tolerance))
                + (-2 * np.arctan2(dx2, arg21 + tolerance))
                - (-2 * np.arctan2(dx1, arg26 + tolerance))
                + (-2 * np.arctan2(dx1, arg31 + tolerance))
                - (-2 * np.arctan2(dx2, arg36 + tolerance))
            )
            rows["bx"][0, nC : 2 * nC] = (
                np.log(arg5)
                - np.log(arg10)
                + np.log(arg15)
                - np.log(arg20)
                + np.log(arg25)
                - np.log(arg30)
                + np.log(arg35)
                - np.log(arg40)
            )
            rows["bx"][0, 2 * nC :] = (
                (np.log(arg4) - np.log(arg9))
                + (np.log(arg14) - np.log(arg19))
                + (np.log(arg24) - np.log(arg29))
                + (np.log(arg34) - np.log(arg39))
            )
            rows["bx"] /= -4 * np.pi

            rows["bx"] *= self.M

        if ("by" in components) or ("tmi" in components):
            rows["by"] = np.zeros((1, 3 * nC))

            rows["by"][0, 0:nC] = (
                np.log(arg5)
                - np.log(arg10)
                + np.log(arg15)
                - np.log(arg20)
                + np.log(arg25)
                - np.log(arg30)
                + np.log(arg35)
                - np.log(arg40)
            )
            rows["by"][0, nC : 2 * nC] = (
                (-2 * np.arctan2(dy2, arg2 + tolerance))
                - (-2 * np.arctan2(dy2, arg7 + tolerance))
                + (-2 * np.arctan2(dy2, arg12 + tolerance))
                - (-2 * np.arctan2(dy2, arg17 + tolerance))
                + (-2 * np.arctan2(dy1, arg22 + tolerance))
                - (-2 * np.arctan2(dy1, arg27 + tolerance))
                + (-2 * np.arctan2(dy1, arg32 + tolerance))
                - (-2 * np.arctan2(dy1, arg37 + tolerance))
            )
            rows["by"][0, 2 * nC :] = (
                (np.log(arg3) - np.log(arg8))
                + (np.log(arg13) - np.log(arg18))
                + (np.log(arg23) - np.log(arg28))
                + (np.log(arg33) - np.log(arg38))
            )

            rows["by"] /= -4 * np.pi

            rows["by"] *= self.M

        if ("bz" in components) or ("tmi" in components):
            rows["bz"] = np.zeros((1, 3 * nC))

            rows["bz"][0, 0:nC] = (
                np.log(arg4)
                - np.log(arg9)
                + np.log(arg14)
                - np.log(arg19)
                + np.log(arg24)
                - np.log(arg29)
                + np.log(arg34)
                - np.log(arg39)
            )
            rows["bz"][0, nC : 2 * nC] = (
                (np.log(arg3) - np.log(arg8))
                + (np.log(arg13) - np.log(arg18))
                + (np.log(arg23) - np.log(arg28))
                + (np.log(arg33) - np.log(arg38))
            )
            rows["bz"][0, 2 * nC :] = (
                (-2 * np.arctan2(dz2, arg1_ + tolerance))
                - (-2 * np.arctan2(dz2, arg6_ + tolerance))
                + (-2 * np.arctan2(dz1, arg11_ + tolerance))
                - (-2 * np.arctan2(dz1, arg16_ + tolerance))
                + (-2 * np.arctan2(dz2, arg21_ + tolerance))
                - (-2 * np.arctan2(dz2, arg26_ + tolerance))
                + (-2 * np.arctan2(dz1, arg31_ + tolerance))
                - (-2 * np.arctan2(dz1, arg36_ + tolerance))
            )
            rows["bz"] /= -4 * np.pi

            rows["bz"] *= self.M

        if "tmi" in components:

            rows["tmi"] = np.dot(
                self.tmi_projection, np.r_[rows["bx"], rows["by"], rows["bz"]]
            )

        return np.vstack([rows[component] for component in components])

    @property
    def deleteTheseOnModelUpdate(self):
        deletes = super().deleteTheseOnModelUpdate
        if self.is_amplitude_data:
            deletes = deletes + ["_gtg_diagonal"]
        return deletes

    @property
    def coordinate_system(self):
        raise AttributeError(
            "The coordinate_system property has been removed. "
            "Instead make use of `SimPEG.maps.SphericalSystem`."
        )


class Simulation3DDifferential(BaseMagneticPDESimulation):
    """
    Secondary field approach using differential equations!
    """

    survey = properties.Instance("a survey object", Survey, required=True)

    storeJ = properties.Bool("store the sensitivity matrix?", default=False)

    _Jmatrix = None

    def __init__(self, mesh,Bs='None', **kwargs):
        super().__init__(mesh, **kwargs)


        Dface = self.mesh.faceDiv

        self.Bs = Bs

        Mc = sdiag(self.mesh.vol)

        if self.mesh._meshType == 'TENSOR':
            Pbc, Pin, self._Pout = self.mesh.getBCProjWF("neumann", discretization="CC")
            self.Dface = Dface * Pin.T * Pin


        else:
            t=0
            face_bound_tot = np.empty(0,dtype=int)
            for face_bound in mesh.faceBoundaryInd:
                if t>1:
                    face_bound = face_bound+mesh.n_faces_x
                if t>3:
                    face_bound = face_bound+mesh.n_faces_y
                t+=1
                face_bound_tot = np.concatenate((face_bound_tot,face_bound))

            self.face_bound_tot = face_bound_tot

            if Bs == 'None':
                self.grhs = 0

            else:
                Bs = self.Bs[face_bound_tot]
                self.Bs=np.zeros_like(self.Bs)
                self.Bs[face_bound_tot] = Bs

                self.grhs = Mc * Dface * self.Bs


            Dface_data = Dface.data
            Dface_ind = Dface.indices
            Dface_ptr = Dface.indptr
            test =npi.indices(Dface_ind,face_bound_tot)
            Dface_data[test] = 0.0

            Dfacetest = sp.csr_matrix((Dface_data,Dface_ind,Dface_ptr),Dface.shape)
            self.Dface = Dfacetest



            '''
            Dface_ptr = Dface.indptr
            #self.grhs = 0


            Dface[:,face_bound_tot]=0
            Dface = sp.csr_matrix(Dface)
            self.Dface = Dface
            '''


        self._Div = Mc * self.Dface


    def makeMassMatrices(self, m):

        mui = self.muiMap * m
        self._MfMuI = self.mesh.getFaceInnerProduct(mui,invert_matrix = True) #/ self.mesh.dim
        self._MfMu0 = self.mesh.getFaceInnerProduct(1.0 / mu_0) #/ self.mesh.dim


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
        """

        .. math ::

            \mathbf{rhs} = \Div(\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0 - \Div\mathbf{B}_0+\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}

        """
        B0 = self.getB0()

        g = self.grhs

        return (self._Div * self.MfMuiI * self.MfMui0 * B0 - self._Div * B0+g)

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div * self.MfMuiI * self._Div.T

    def fields(self, m):
        """
        Return magnetic potential (u) and flux (B)
        u: defined on the cell center [nC x 1]
        B: defined on the cell center [nG x 1]

        After we compute u, then we update B.

        .. math ::

            \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        self.model = m  # / self.mesh.dim
        self.MfMui0 = self.mesh.getFaceInnerProduct(1.0 / mu_0)


        A = self.getA(m)
        rhs = self.getRHS(m)
        Ainv = self.solver(A, **self.solver_opts)

        u = Ainv * rhs
        B0 = self.getB0()
        B = self.MfMui0 * self.MfMuiI * B0 - B0 - self.MfMuiI * self._Div.T * u

        Ainv.clean()

        return {"B": B, "u": u}

    def dpred(self, m=None, f=None):

        if f is None:
            f = self.fields(m)

        dpred = self.projectFields(f)
        return(dpred)

    def Jvec(self, m, v, f=None):

        if self.storeJ:
            J = self.getJ(m, f=f)
            return J.dot(v)

        self.model = m
        if f is None:
            f = self.fields(m)

        B, u = f["B"], f["u"]

        Q = self.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        C = -self.MfMuiI * self._Div.T

        MfMuiIderiv = self.MfMuiIDeriv
        dMfMuiI_divT_u_v = MfMuiIderiv(self._Div.T @ u, v, adjoint=False)

        MfMui0_at_dMfMuiI_B0_v = self.MfMui0 * MfMuiIderiv(B0, v, adjoint=False)

        dq_dm_min_dAmu_dm = self._Div*(MfMui0_at_dMfMuiI_B0_v-dMfMuiI_divT_u_v)
        db_dm = MfMui0_at_dMfMuiI_B0_v
        dCmu_dm = -dMfMuiI_divT_u_v


        A = self.getA(m)  # = A

        Ainv = self.solver(A, **self.solver_opts)
        sol = Ainv * (dq_dm_min_dAmu_dm)
        test = C*sol

        Jv = Q*(test + (dCmu_dm + db_dm))

        Ainv.clean()
        Jv = mkvc(Jv)

        return Jv


    def Jtvec(self, m, v, f=None):


        if self.storeJ:
            J = self.getJ(m, f=f)
            return np.asarray(J.T.dot(v))

        self.model = m

        if f is None:
            f = self.fields(m)

        B, u = f["B"], f["u"]

        Q = self.projectFieldsDeriv(B)
        B0 = self.getB0()

        MfMuiIderiv = self.MfMuiIDeriv
        vtest = -self._Div*self.MfMuiI.T*Q.T*v
        A_T = self.getA(m)
        Ainv_T = self.solver(A_T, **self.solver_opts)
        sol = Ainv_T * vtest

        Ainv_T.clean()

        DivTatsol = self._Div.T * sol
        DivTatu = self._Div.T * u

        dCmu_min_dAmu = MfMuiIderiv(DivTatu, Q.T*v - DivTatsol, adjoint=True)
        dq_dm_plus_db_dm = MfMuiIderiv(B0, self.MfMui0.T * (DivTatsol + Q.T*v), adjoint=True)

        Jtv = dCmu_min_dAmu + dq_dm_plus_db_dm

        return Jtv

    def getJ(self, m, f=None):

        if self._Jmatrix is None:

            self.model = m

            if f is None:
                f = self.fields(m)

            B, u = f["B"], f["u"]

            Q = self.projectFieldsDeriv(B)
            B0 = self.getB0()

            MfMuiIderiv = self.MfMuiIDeriv
            vtest = (-self._Div*self.MfMuiI.T*Q.T).toarray()
            A_T = self.getA(m)
            Ainv_T = self.solver(A_T, **self.solver_opts)
            sol = Ainv_T * vtest

            Ainv_T.clean()

            DivTatsol = self._Div.T * sol
            DivTatu = self._Div.T * u

            dCmu_min_dAmu = MfMuiIderiv(DivTatu,Q.T.toarray()-DivTatsol,adjoint = True)
            dq_dm_plus_db_dm = MfMuiIderiv(B0, self.MfMui0.T*(DivTatsol+Q.T.toarray()), adjoint=True)

            Jtv = dCmu_min_dAmu + dq_dm_plus_db_dm

        else:
            Jtv = self._Jmatrix.T

        if self.storeJ == True:
            self._Jmatrix = Jtv.T

        return Jtv.T



    @property
    def Qfx(self):
        if getattr(self, "_Qfx", None) is None:
            self._Qfx = self.mesh.getInterpolationMat(
                self.survey.receiver_locations, "Fx"
            )
        return self._Qfx

    @property
    def Qfy(self):
        if getattr(self, "_Qfy", None) is None:
            self._Qfy = self.mesh.getInterpolationMat(
                self.survey.receiver_locations, "Fy"
            )
        return self._Qfy

    @property
    def Qfz(self):
        if getattr(self, "_Qfz", None) is None:
            self._Qfz = self.mesh.getInterpolationMat(
                self.survey.receiver_locations, "Fz"
            )
        return self._Qfz

    def projectFields(self, u):
        """
        This function projects the fields onto the data space.
        Especially, here for we use total magnetic intensity (TMI) data,
        which is common in practice.
        First we project our B on to data location

        .. math::

            \mathbf{B}_{rec} = \mathbf{P} \mathbf{B}

        then we take the dot product between B and b_0

        .. math ::

            \\text{TMI} = \\vec{B}_s \cdot \hat{B}_0

        """
        # TODO: There can be some different tyes of data like |B| or B
        components = self.survey.components

        fields = {}
        '''
        fields["bx"] = self.Qfx * B
        fields["by"] = self.Qfy * B
        fields["bz"] = self.Qfz * B

        '''
        if "bx" in components or "tmi" in components:
            fields["bx"] = self.Qfx * u["B"]
        if "by" in components or "tmi" in components:
            fields["by"] = self.Qfy * u["B"]
        if "bz" in components or "tmi" in components:
            fields["bz"] = self.Qfz * u["B"]

        if "tmi" in components:
            bx = fields["bx"]
            by = fields["by"]
            bz = fields["bz"]
            # Generate unit vector
            B0 = self.survey.source_field.b0
            Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)
            box = B0[0] / Bot
            boy = B0[1] / Bot
            boz = B0[2] / Bot
            fields["tmi"] = bx * box + by * boy + bz * boz

        return np.concatenate([fields[comp] for comp in components])

    @utils.count
    def projectFieldsDeriv(self, B):
        """
        This function projects the fields onto the data space.

        .. math::

            \\frac{\partial d_\\text{pred}}{\partial \mathbf{B}} = \mathbf{P}

        Especially, this function is for TMI data type
        """

        components = self.survey.components

        fields = {}

        if "bx" in components or "tmi" in components:
            fields["bx"] = self.Qfx
        if "by" in components or "tmi" in components:
            fields["by"] = self.Qfy
        if "bz" in components or "tmi" in components:
            fields["bz"] = self.Qfz

        if "tmi" in components:
            bx = fields["bx"]
            by = fields["by"]
            bz = fields["bz"]
            # Generate unit vector
            B0 = self.survey.source_field.b0
            Bot = np.sqrt(B0[0] ** 2 + B0[1] ** 2 + B0[2] ** 2)
            box = B0[0] / Bot
            boy = B0[1] / Bot
            boz = B0[2] / Bot
            fields["tmi"] = bx * box + by * boy + bz * boz

        return sp.vstack([fields[comp] for comp in components])

    def projectFieldsAsVector(self, B):

        bfx = self.Qfx * B
        bfy = self.Qfy * B
        bfz = self.Qfz * B


        return np.r_[bfx, bfy, bfz]

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super().deleteTheseOnModelUpdate
        if self._Jmatrix is not None:
            toDelete = toDelete + ["_Jmatrix"]
        return toDelete


def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):
    """
    Inversion module for MagneticsDiffSecondary

    """
    from SimPEG import Optimization, Regularization, Parameters, ObjFunction, Inversion

    prob = Simulation3DDifferential(mesh, survey=data, mu=model)

    miter = kwargs.get("maxIter", 10)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=miter)
    opt.bfgsH0 = Solver(sp.identity(model.nP), flag="D")
    # Create a regularization program
    reg = Regularization.Tikhonov(model)
    # Create an objective function
    beta = Parameters.BetaSchedule(beta0=1e0)
    obj = ObjFunction.BaseObjFunction(prob, reg, beta=beta)
    # Create an inversion object
    inv = Inversion.BaseInversion(obj, opt)

    return inv, reg
