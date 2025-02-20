import re
from base64 import b64decode
from os import path
import numpy as np
import pandas as pd
from lxml import objectify
from collections import OrderedDict
from pylinac.core.profile import Normalization
from .wtscans import XyProfile, PDD
from .detector_array import DetectorArray
from .gamma_analysis import Gamma


def get_xcc_objects(file):
    xml = objectify.parse(open(file))
    root = xml.getroot()
    root.getchildren()[0].getchildren()
    return root.getchildren()[4].getchildren()


class ReadMCC:
    _tags = ['MODALITY', 'ISOCENTER', 'ENERGY', 'SSD', 'FIELD_INPLANE', 'FIELD_CROSSPLANE', 'GANTRY', 'COLL_ANGLE',
             'FILTER', 'SCAN_CURVETYPE', 'SCAN_DEPTH', 'SCAN_OFFAXIS_INPLANE', 'SCAN_OFFAXIS_CROSSPLANE', 'DETECTOR',
             'CORRECTIONS', 'COMMENT']
    _oct_1500_detectors = [27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27,
                           26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26, 27, 26,
                           27, 26, 27, 26, 27, 26, 27]
    get_all_depths = False

    def __init__(self, file: str, **kwargs: object) -> None:
        """
            Reads a scan file, mcc, xcc for water tank or diode arrays.

            Parameters
            ----------
            file : str
                File name.

            kwargs:
                depths (list): List of profile depths for analysis.
                ion_to_dose (bool): Convert depth ionisation to depth dose.
                normalisation_pdd (bool): Normalise the PDD.
                normalisation_profile (Normalization): Pylinac Normalization object

        """
        self.file = file

        if 'depths' in kwargs:
            depths = kwargs.pop('depths')
            if depths == 'all':
                depths = self.get_depths()

        else:
            depths = [100.00]
        if 'ion_to_dose' in kwargs:
            ion_to_dose = kwargs.pop('ion_to_dose')
        else:
            ion_to_dose = False
        if 'normalise_pdd' in kwargs:
            normalise_pdd = kwargs.pop('normalise_pdd')
        else:
            normalise_pdd = True
        if 'normalise_profile' in kwargs:
            normalise_profile = kwargs.pop('normalise_profile')
        else:
            normalise_profile = Normalization.BEAM_CENTER
        if 'array_profiles' in kwargs:
            array_profiles = kwargs.pop('array_profiles')
        else:
            array_profiles = False
        if 'energy' in kwargs:
            energy = kwargs.pop('energy')
        else:
            energy = None
        if 'scale' in kwargs:
            scale = kwargs.pop('scale')
        else:
            scale = 20
        if 'smoothing_factor' in kwargs:
            smoothing_factor = kwargs.pop('smoothing_factor')
        else:
            smoothing_factor = None

        meta_data = {}
        clean = {}
        data = []
        file_type = path.splitext(file)[1][1:]

        if file_type == 'mcc':
            scans = self.get_scans(file)
            scan_data = self.separate_data(scans)
            clean = self.clean_data(scan_data)
            # mid = int(len(meta_data) / 2)
            meta_data = self.get_metadata(scans)

            data = meta_data
        elif file_type == 'xcc':
            root = get_xcc_objects(file)
            clean = self.get_xcc_data(root)
            meta_data = self.get_xcc_meta_data(root)

        _mcc_data = {}
        if array_profiles:
            if file_type == 'mcc':
                detectors = DetectorArray(meta_data[0], clean, normalise_profile=normalise_profile, scale=scale,
                                          smoothing_factor=smoothing_factor)
            else:
                detectors = DetectorArray(meta_data, clean, normalise_profile=normalise_profile, scale=scale,
                                          smoothing_factor=smoothing_factor)
            _mcc_data['INPLANE_PROFILE'] = detectors.inplane
            _mcc_data['CROSSPLANE_PROFILE'] = detectors.crossplane
            _mcc_data['PROFILE_GRID'] = detectors.grid
        else:
            for i in range(0, len(meta_data)):
                dataset = pd.DataFrame({'Position': clean[i][:, 0], 'Values': clean[i][:, 1]})
                if not array_profiles:
                    dataset = self.up_sample(dataset, scale)
                data[i].append(dataset)

                if data[i][0] == 'PDD':
                    if energy:  # Added for electrons where all energies are in one file
                        if data[i][1]['ENERGY'] == energy:
                            _mcc_data[data[i][0]] = PDD(data[i], normalise_pdd, ion_to_dose).results
                    else:
                        _mcc_data[data[i][0]] = PDD(data[i], normalise_pdd, ion_to_dose).results
                if data[i][0] == 'INPLANE_PROFILE':
                    if data[i][1]['SCAN_DEPTH'] in depths:
                        _mcc_data[data[i][0] + '_' + str(data[i][1]['SCAN_DEPTH'])] = XyProfile(data[i], normalise_profile).results
                if data[i][0] == 'CROSSPLANE_PROFILE':
                    if data[i][1]['SCAN_DEPTH'] in depths:
                        _mcc_data[data[i][0] + '_' + str(data[i][1]['SCAN_DEPTH'])] = XyProfile(data[i], normalise_profile).results
        # Ensure order PDD, inplane and crossplane
        self.mcc_data = OrderedDict(sorted(_mcc_data.items(), reverse=True))
        pass

    def get_depths(self):
        depths = []
        with open(self.file, "r") as fi:
            id = []
            for ln in fi:
                line = ln.strip()
                if line.startswith("SCAN_DEPTH"):
                    line = line.split('=')
                    depths.append(line[1])
        return list(set(depths))


    @staticmethod
    def get_xcc_meta_data(root):
        data = {}
        acc = root[3].getchildren()

        scan_depth = root[6].getchildren()
        data['ENERGY'] = acc[0].pyval
        data['SCAN_DEPTH'] = scan_depth[0].pyval
        data['FIELD_CROSSPLANE'] = acc[2].pyval
        data['FIELD_INPLANE'] = acc[3].pyval
        data['SSD'] = acc[4].pyval
        data['FILTER'] = acc[5].pyval.split('_')[2]
        data['GANTRY'] = root[9].getchildren()[0].getchildren()[0].pyval
        data['SCAN_CURVETYPE'] = 'CROSSPLANE_PROFILE'
        data['ISOCENTER'] = 0
        data['MODALITY'] = root[4].getchildren()[1].pyval.replace('PTW_MODALITY_PHOTONS', 'X')
        data['COLL_ANGLE'] = acc[1].pyval
        data['DETECTOR'] = root[1].getchildren()[4].pyval.replace('PTW_DETECTOR_', '')
        data['SCAN_OFFAXIS_INPLANE'] = None
        data['SCAN_OFFAXIS_CROSSPLANE'] = None
        data['COMMENT'] = root[5].getchildren()[0].pyval
        return ['CROSSPLANE_PROFILE', data]

    def get_xcc_data(self, root):
        meas = root[9].getchildren()
        data = []
        for i, m in enumerate(meas):
            coded_string = str(m[i].getchildren()[2])
            decode = b64decode(coded_string)
            data.append(np.frombuffer(decode, dtype=np.float32))
            pass
        data = np.asarray(data)
        data_sum = np.sum(data, axis=0)

        arr = []
        start = 0
        for det in self._oct_1500_detectors:
            pos = np.linspace(-130, 130, det)
            stop = det + start
            s = data_sum[start:stop]
            arr.append(np.transpose(np.array([pos, s])))
            start = stop
        return arr

    @staticmethod
    def get_scans(dir_path: str) -> list:
        """
        Extracts all the data between the BEGIN_SCAN and END_SCAN of each scan within the mcc file.
        Each scan is stored in a list.

        Parameters
        ----------
        dir_path : str
            File name.

        Returns
        -------
        scans : list
            A list of each scan present in the PTW mmc file.

        """
        start_pattern = '^BEGIN_SCAN.*$'
        end_pattern = '^END_SCAN.*$'
        scan = []
        scans = []
        with open(dir_path) as file:
            match = False
            for i, line in enumerate(file):
                if i != 0:
                    line = line.strip()
                    if re.match(start_pattern, line):
                        match = True
                        continue
                    elif re.match(end_pattern, line):
                        match = False
                        scans.append(scan)
                        scan = []
                        continue
                    elif match:
                        if not line.startswith('REF_SCAN_POSITIONS'):
                            scan.append(line)

        scans = list(filter(None, scans))
        return scans

    @staticmethod
    def separate_data(data: list) -> list:
        """
        Extracts all the position/output data between the BEGIN_DATA and END_DATA of each scan within the mcc file.
        Each scan is stored in a list.

        Parameters
        ----------
        data : list
            A list of each scan present in the PTW mmc file.

        Returns
        -------
        scans_data : list
            A list of each scan's position and output data.

        """
        start_pattern = '^BEGIN_DATA.*$'
        end_pattern = '^END_DATA.*$'
        scan_data = []
        scans_data = []
        match = False
        for dd in data:
            for i, line in enumerate(dd):
                line = line.strip()
                if re.match(start_pattern, line):
                    match = True
                    continue
                elif re.match(end_pattern, line):
                    match = False
                    scans_data.append(scan_data)
                    scan_data = []
                    continue
                elif match:
                    scan_data.append(line)

        scans_data = list(filter(None, scans_data))
        return scans_data

    @staticmethod
    def clean_data(data: list) -> list:
        """
        Converts each line of position/output data from double tab separated into a numpy array.

        Parameters
        ----------
        data : list
            A list of each scan present in the PTW mmc file.

        Returns
        -------
        scans_data : list
            A list of each scan's position and output data.

        """
        scan_data = []
        scans_data = []
        for dd in data:
            for line in dd:
                scan_data.append(line.split('\t\t')[:2])
            scans_data.append(np.asarray(scan_data).astype(np.float32))
            scan_data = []
        return scans_data

    def get_metadata(self, scans: list) -> list:
        """
        Extracts each scan's metadata.

        Parameters
        ----------
        scans : list
            A list of metadata present in the PTW mmc file.

        Returns
        -------
        meta_data : list
            A list of each scan's metadata.

        """
        meta = {}
        meta_data = []
        data_type = None
        matching_scans = []
        for scan in scans:
            matching = [s for s in scan if any(xs in s for xs in self._tags)]
            matching_scans.append(matching)
        for match in matching_scans:
            for line in match:
                if 'SCAN_CURVETYPE' in line:
                    data_type = line.split('=')[1]
                name = line.split('=')[0]
                try:
                    value = float(line.split('=')[1])
                except ValueError:
                    value = line.split('=')[1]
                meta[name] = value
            meta_data.append([data_type, meta])
            meta = {}
        return meta_data

    @staticmethod
    def up_sample(dataframe: pd.DataFrame, factor) -> pd.DataFrame:
        """
        Calculate an interpolated "x" value between two points.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Pandas dataframe of positions and output.
        factor : int
            Scaling factor, default 10.

        Returns
        -------
        data_interp : pd.DataFrame
            Dataframe of positions and output data scaled..

        """
        # ndarray with 10th mm resolution between first and last measurement
        # position, beginning at the first index (could be smaller than 0!)

        start = dataframe.Position[dataframe.first_valid_index()]  # * factor
        # last index
        stop = dataframe.Position[dataframe.last_valid_index()]  # * factor

        interp_arr = np.linspace(start, stop, 53 * factor)

        # create dict with single entry to construct dataframe with column names
        df_interp = {"Position": interp_arr}
        # convert dict to 1-D dataframe
        data_interp = pd.DataFrame(df_interp)

        # connect data frames
        data_interp = pd.merge_ordered(data_interp, dataframe, on="Position", how="outer")
        data_interp.Values = data_interp.Values.interpolate()

        return data_interp

    @staticmethod
    def calc_gamma(reference: object, evaluation: object, **kwargs) -> dict:
        """
            Calculate an interpolated "x" value between two points.

            Parameters
            ----------
            reference : ReadMCC
                Instance of ReadMCC, reference data.
            evaluation : ReadMCC
                Instance of ReadMCC, evaluation data.

            Returns
            -------
            gamma_analysis : dict
                Dictionary of gamma values.

                """
        if reference and evaluation:
            _reference = reference
            _evaluation = evaluation

            gamma_analysis = {}
            for (key_ref, val_ref), (key_eval, val_eval) in zip(_reference.items(), _evaluation.items()):
                gamma_analysis[key_ref] = Gamma([key_ref, val_ref], [key_eval, val_eval],
                                                **kwargs)
            return gamma_analysis
