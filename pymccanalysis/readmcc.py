import re
from pathlib import Path
from base64 import b64decode
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lxml import objectify
from pylinac.core.profile import Normalization

from .wtscans import XyProfile, PDD
from .detector_array import DetectorArray
from .gamma_analysis import Gamma


def get_xcc_objects(file_path: str) -> Any:
    """
    Parse an XCC file and return scan-related XML objects.

    Parameters
    ----------
    file_path : str
        Path to the .xcc file.

    Returns
    -------
    Any
        XML child objects containing scan data.
    """
    with open(file_path, "rb") as f:
        xml = objectify.parse(f)

    root = xml.getroot()
    return root.getchildren()[4].getchildren()


class ReadMCC:
    """
    Parser and processor for MCC/XCC scan files.

    Handles extraction of metadata, decoding of detector data,
    interpolation, and generation of profiles/PDDs.
    """

    _tags: set[str] = {
        'MODALITY', 'ISOCENTER', 'ENERGY', 'SSD',
        'FIELD_INPLANE', 'FIELD_CROSSPLANE', 'GANTRY',
        'COLL_ANGLE', 'FILTER', 'SCAN_CURVETYPE',
        'SCAN_DEPTH', 'SCAN_OFFAXIS_INPLANE',
        'SCAN_OFFAXIS_CROSSPLANE', 'DETECTOR',
        'CORRECTIONS', 'COMMENT'
    }

    _oct_1500_detectors: List[int] = [
        27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26,
        27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26,
        27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26,
        27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26,
        27, 26, 27, 26, 27
    ]

    mcc_data: Optional[Dict[str, Any]] = None

    def __init__(self, file: str, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        file : str
            Path to .mcc or .xcc file.

        kwargs
        ------
        depths : list[float] | str
            Depth values to include or 'all'.
        ion_to_dose : bool
            Convert ionization to dose if True.
        normalise_pdd : bool
            Normalize PDD curves.
        normalise_profile : Normalization
            Profile normalization mode.
        array_profiles : bool
            Whether to treat input as detector array.
        energy : Optional[str]
            Energy filter for PDD.
        scale : int
            Upsampling factor.
        smoothing_factor : Optional[float]
            Optional smoothing applied to profiles.
        """

        self.file_type: str = Path(file).suffix.lower()

        if self.file_type not in {'.mcc', '.xcc'}:
            raise OSError("File must be .mcc or .xcc")

        self.file: str = file

        self.depths: Union[List[float], str] = kwargs.pop('depths', [100.00])
        if self.depths == 'all':
            self.depths = self.get_depths()

        self.ion_to_dose: bool = kwargs.pop('ion_to_dose', False)
        self.normalise_pdd: bool = kwargs.pop('normalise_pdd', True)

        self.normalise_profile: Normalization = kwargs.pop(
            'normalise_profile',
            Normalization.BEAM_CENTER
        )

        self.array_profiles: bool = kwargs.pop('array_profiles', False)
        self.energy: Optional[str] = kwargs.pop('energy', None)
        self.scale: int = kwargs.pop('scale', 10)
        self.smoothing_factor: Optional[float] = kwargs.pop(
            'smoothing_factor',
            None
        )

        self.meta_data, self.clean = self.extract_data()

    # -------------------------------------------------------------------------
    # MAIN PROCESSING
    # -------------------------------------------------------------------------

    def process_data(self) -> None:
        """
        Process extracted scan data into profiles and/or PDD results.

        Populates:
            self.mcc_data (OrderedDict)
        """
        processed: Dict[str, Any] = {}

        if self.array_profiles:
            detector_input = (
                self.meta_data[0]
                if self.file_type == '.mcc'
                else self.meta_data
            )

            detectors = DetectorArray(
                detector_input,
                self.clean,
                normalise_profile=self.normalise_profile,
                scale=self.scale,
                smoothing_factor=self.smoothing_factor
            )

            processed['INPLANE_PROFILE'] = detectors.inplane
            processed['CROSSPLANE_PROFILE'] = detectors.crossplane
            processed['PROFILE_GRID'] = detectors.grid

        else:
            for meta, clean in zip(self.meta_data, self.clean):
                scan_type, metadata = meta

                dataset = self.up_sample_numpy(clean, self.scale)
                meta.append(dataset)

                if scan_type == 'PDD':
                    if self.energy is None or metadata.get('ENERGY') == self.energy:
                        processed[scan_type] = PDD(
                            meta,
                            self.normalise_pdd,
                            self.ion_to_dose
                        ).results

                elif scan_type in ('INPLANE_PROFILE', 'CROSSPLANE_PROFILE'):
                    depth = metadata.get('SCAN_DEPTH')

                    if depth in self.depths:
                        key = f"{scan_type}_{round(depth)}"
                        processed[key] = XyProfile(
                            meta,
                            self.normalise_profile
                        ).results

        self.mcc_data = OrderedDict(sorted(processed.items(), reverse=True))

    # -------------------------------------------------------------------------
    # FILE EXTRACTION
    # -------------------------------------------------------------------------

    def extract_data(self) -> Tuple[List[Any], List[Any]]:
        """
        Extract metadata and cleaned scan data from file.

        Returns
        -------
        tuple
            (meta_data, clean_data)
        """
        if self.file_type == '.mcc':
            scans = self.get_scans(self.file)
            scan_data = self.separate_data(scans)
            clean = self.clean_data(scan_data)
            meta_data = self.get_metadata(scans)
            return meta_data, clean

        elif self.file_type == '.xcc':
            root = get_xcc_objects(self.file)
            clean = self.get_xcc_data(root)
            meta_data = self.get_xcc_meta_data(root)
            return meta_data, clean

        raise ValueError("Unsupported file type")

    # -------------------------------------------------------------------------
    # DEPTHS
    # -------------------------------------------------------------------------

    def get_depths(self) -> List[str]:
        """
        Extract scan depth values from MCC file.

        Returns
        -------
        list[str]
        """
        with open(self.file, "r") as fi:
            return list({
                line.split('=', 1)[1].strip()
                for line in fi
                if line.startswith("SCAN_DEPTH")
            })

    # -------------------------------------------------------------------------
    # XCC PARSING
    # -------------------------------------------------------------------------

    @staticmethod
    def get_xcc_meta_data(root: Any) -> List[Any]:
        """
        Extract metadata from XCC XML structure.
        """
        data: Dict[str, Any] = {}

        acc = root[3].getchildren()
        scan_depth = root[6].getchildren()

        data['ENERGY'] = acc[0].pyval
        data['SCAN_DEPTH'] = scan_depth[0].pyval
        data['FIELD_CROSSPLANE'] = acc[2].pyval
        data['FIELD_INPLANE'] = acc[3].pyval
        data['SSD'] = acc[4].pyval
        data['FILTER'] = acc[5].pyval.split('_')[2]

        data['GANTRY'] = (
            root[9]
            .getchildren()[0]
            .getchildren()[0]
            .pyval
        )

        data['SCAN_CURVETYPE'] = 'CROSSPLANE_PROFILE'
        data['ISOCENTER'] = 0

        data['MODALITY'] = (
            root[4]
            .getchildren()[1]
            .pyval
            .replace('PTW_MODALITY_PHOTONS', 'X')
        )

        data['COLL_ANGLE'] = acc[1].pyval

        data['DETECTOR'] = (
            root[1]
            .getchildren()[4]
            .pyval
            .replace('PTW_DETECTOR_', '')
        )

        data['SCAN_OFFAXIS_INPLANE'] = None
        data['SCAN_OFFAXIS_CROSSPLANE'] = None

        data['COMMENT'] = root[5].getchildren()[0].pyval

        return ['CROSSPLANE_PROFILE', data]

    def get_xcc_data(self, root: Any) -> List[np.ndarray]:
        """
        Decode detector data from XCC file.

        Returns
        -------
        list[np.ndarray]
        """
        meas = root[9].getchildren()

        decoded_arrays = [
            np.frombuffer(
                b64decode(str(m[i].getchildren()[2])),
                dtype=np.float32
            )
            for i, m in enumerate(meas)
        ]

        data_sum = np.sum(decoded_arrays, axis=0)
        arr: List[np.ndarray] = []
        start = 0

        for det in self._oct_1500_detectors:
            stop = start + det
            pos = np.linspace(-130, 130, det)
            values = data_sum[start:stop]

            arr.append(np.column_stack((pos, values)))
            start = stop

        return arr

    # -------------------------------------------------------------------------
    # MCC PARSING
    # -------------------------------------------------------------------------

    @staticmethod
    def get_scans(file_path: str) -> List[List[str]]:
        """
        Extract raw scan blocks from MCC file.
        """
        scans: List[List[str]] = []
        scan: List[str] = []
        inside_scan = False

        with open(file_path) as file:
            next(file, None)

            for line in file:
                line = line.strip()

                if line.startswith("BEGIN_SCAN"):
                    inside_scan = True
                    continue

                if line.startswith("END_SCAN"):
                    inside_scan = False

                    if scan:
                        scans.append(scan)

                    scan = []
                    continue

                if inside_scan and not line.startswith("REF_SCAN_POSITIONS"):
                    scan.append(line)

        return scans

    @staticmethod
    def separate_data(data: List[List[str]]) -> List[List[str]]:
        """
        Separate scan metadata from scan data blocks.
        """
        scans_data: List[List[str]] = []

        for scan in data:
            scan_data: List[str] = []
            inside_data = False

            for line in scan:
                if line.startswith("BEGIN_DATA"):
                    inside_data = True
                    continue

                if line.startswith("END_DATA"):
                    inside_data = False

                    if scan_data:
                        scans_data.append(scan_data)

                    scan_data = []
                    continue

                if inside_data:
                    scan_data.append(line)

        return scans_data

    @staticmethod
    def clean_data(data: List[List[str]]) -> List[np.ndarray]:
        """
        Convert raw scan strings into numeric numpy arrays.
        """
        scans_data: List[np.ndarray] = []

        for scan in data:
            arr = np.array([line.split('\t\t')[:2] for line in scan], dtype=np.float32)
            scans_data.append(arr)

        return scans_data

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    def get_metadata(self, scans: List[List[str]]) -> List[List[Any]]:
        """
        Extract metadata dictionaries from scans.
        """
        meta_data: List[List[Any]] = []

        for scan in scans:
            meta: Dict[str, Any] = {}
            data_type: Optional[str] = None

            for line in scan:
                if '=' not in line:
                    continue

                name, value = line.split('=', 1)

                if name not in self._tags:
                    continue

                try:
                    value = float(value)
                except ValueError:
                    pass

                if name == 'SCAN_CURVETYPE':
                    data_type = value

                meta[name] = value

            meta_data.append([data_type, meta])

        return meta_data

    # -------------------------------------------------------------------------
    # INTERPOLATION
    # -------------------------------------------------------------------------

    @staticmethod
    def up_sample_numpy(data: np.ndarray, factor: int) -> pd.DataFrame:
        """
        Upsample a 2-column numpy array using linear interpolation.

        Returns
        -------
        pd.DataFrame
        """
        x = data[:, 0]
        y = data[:, 1]
        length = len(x)

        if x[0] >= 0:
            interp_x = np.linspace(x[0], x[-1], num=length * factor)
        else:
            a = np.linspace(x[0], 0, num=length * factor, endpoint=False)
            b = np.linspace(0, x[-1], num=length * factor + 1)
            interp_x = np.append(a, b)

        interp_y = np.interp(interp_x, x, y)

        return pd.DataFrame({"Position": interp_x, "Values": interp_y})

    # -------------------------------------------------------------------------
    # GAMMA
    # -------------------------------------------------------------------------

    @staticmethod
    def calc_gamma(reference: Dict[str, Any], evaluation: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Compute gamma analysis for matching datasets.

        Returns
        -------
        dict
            Mapping of dataset keys to Gamma objects.
        """
        if not (reference and evaluation):
            return {}

        matches = reference.keys() & evaluation.keys()

        return {
            key: Gamma(
                [key, reference[key]],
                [key, evaluation[key]],
                **kwargs
            )
            for key in matches
        }