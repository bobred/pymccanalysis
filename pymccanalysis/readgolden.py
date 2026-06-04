import pandas as pd
from os import path
from collections import OrderedDict
from pylinac.core.profile import Normalization
from pymccanalysis.wtscans import XyProfile, PDD


class ReadGoldenData:
    pdd_fields = {3: '3x3cm2', 4: '4x4cm2', 6: '6x6cm2', 8: '8x8cm2', 10: '10x10cm2', 15: '15x15cm2', 20: '20x20cm2',
                  25: '25x25cm2', 30: '30x30cm2', 40: '40x40cm2'}
    cross_fields = {3: 'Field Size: 3x3 cm2', 4: 'Field Size: 4x4 cm2', 6: 'Field Size: 6x6 cm2', 8: 'Field Size: 8x8 cm2',
                    10: 'Field Size: 10x10 cm2', 20: 'Field Size: 20x20 cm2', 30: 'Field Size: 30x30 cm2',
                    40: 'Field Size: 40x40 cm2'}

    def __init__(self, base_dir: str, model: str, energy: str, field_size: int = 10, depth: float = 10, **kwargs) -> None:
        if 'normalise_pdd' in kwargs:
            self.normalise_pdd = kwargs.pop('normalise_pdd')
        else:
            self.normalise_pdd = True
        if 'normalise_profile' in kwargs:
            self.normalise_profile = kwargs.pop('normalise_profile')
        else:
            self.normalise_profile = Normalization.BEAM_CENTER
        file_dir = None
        meta_data = {}
        _mcc_data = {}
        if model.lower() == 'truebeam':
            if energy.lower() == '6x' or energy.lower() == '6mv':
                file_dir = path.join(base_dir, r'6MV Beam Data.xlsx')
                pdd = pd.read_excel(file_dir, engine='openpyxl', sheet_name='Open Field Depth Dose', skiprows=5)
                _mcc_data['PDD'] = self.get_pdd(pdd, 'X', '6.0', 'FF', field_size)

                crossplane = pd.read_excel(file_dir, engine='openpyxl', sheet_name='Open Field Profiles at ' + str(depth) + 'cm', skiprows=7)
                _mcc_data['CROSSPLANE_PROFILE_' + str(depth * 10)] = self.get_crossplane(crossplane, 'X', '6.0', 'FF', field_size, depth)

            if energy.lower() == '10x' or energy.lower() == '10mv':
                file_dir = path.join(base_dir, r'10MV Beam Data.xlsx')
                pdd = pd.read_excel(file_dir, engine='openpyxl', sheet_name='Open Field Depth Dose', skiprows=5)
                _mcc_data['PDD'] = self.get_pdd(pdd, 'X', '10.0', 'FF', field_size)

                crossplane = pd.read_excel(file_dir, engine='openpyxl', sheet_name='Open Field Profiles at ' + str(depth) + 'cm', skiprows=7)
                _mcc_data['CROSSPLANE_PROFILE_' + str(depth * 10)] = self.get_crossplane(crossplane, 'X', '10.0', 'FF', field_size, depth)

            if energy.lower() == '10fff':
                file_dir = path.join(base_dir, r'10FFF Beam Data.xlsx')
                pdd = pd.read_excel(file_dir, engine='openpyxl', sheet_name='Open Field Depth Dose', skiprows=5)
                _mcc_data['PDD'] = self.get_pdd(pdd, 'X', '10.0', 'FFF', field_size)

                crossplane = pd.read_excel(file_dir, engine='openpyxl', sheet_name='Open Field Profiles at ' + str(depth) + 'cm', skiprows=7)
                _mcc_data['CROSSPLANE_PROFILE_' + str(depth * 10)] = self.get_crossplane(crossplane, 'X', '10.0', 'FFF', field_size, depth)

            if energy.lower() == '6mev':
                file_dir = path.join(base_dir, r'Electron Beam Data.xlsx')
                pdd = pd.read_excel(file_dir, engine='openpyxl', sheet_name='6MeV Depth Doses', skiprows=5)
                _mcc_data['PDD'] = self.get_pdd(pdd, 'EL', '6.0', 'FF', field_size)
                pass

        self.mcc_data = OrderedDict(sorted(_mcc_data.items(), reverse=True))
        pass

    def get_crossplane(self, crossplane, modality, energy, filter_type, field_size, depth):
        meta_data = {}
        cross_dataset = pd.DataFrame(
            {'Position': crossplane.iloc[:, 0] * 10, 'Values': crossplane[self.cross_fields[field_size]]})
        meta_data['MODALITY'] = modality
        meta_data['ENERGY'] = energy
        meta_data['FILTER'] = filter_type
        meta_data['FIELD_CROSSPLANE'] = field_size
        meta_data['SSD'] = 100
        meta_data['SCAN_DEPTH'] = depth
        meta_data['ISOCENTER'] = 0
        cross_dataset = cross_dataset[cross_dataset['Values'].notna()]
        return XyProfile(['CROSSPLANE_PROFILE', meta_data, cross_dataset], self.normalise_profile).results


    def get_pdd(self, pdd, modality, energy, filter_type, field_size, ion_to_dose=False):
        meta_data = {}
        pdd_dataset = pd.DataFrame({'Position': pdd.iloc[:, 0] * 10, 'Values': pdd[self.pdd_fields[field_size]]})
        pdd_dataset.dropna(inplace=True)
        meta_data['MODALITY'] = modality
        meta_data['ENERGY'] = energy
        meta_data['FILTER'] = filter_type
        meta_data['FIELD_CROSSPLANE'] = field_size
        return PDD(['PDD', meta_data, pdd_dataset], self.normalise_pdd, ion_to_dose).results


