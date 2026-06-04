import pandas as pd
import matplotlib.pyplot as plt
from pymccanalysis.gamma_analysis import gamma_hist, gamma_curve, Gamma


file = r'X:\Physics\RPU\RayStation\Commissioning\BeamData\LA4\FinalSet_SmoothedOnceAndSymmetrised\10FFF\Golden Data Comparison\Crossplanes\crossplane.xlsx'

depth = '20cm'
#rcht_data = pd.read_excel(file, usecols="u:v", sheet_name=depth)
#rcht_data = pd.read_excel(file, usecols="w:x", sheet_name=depth)
#rcht_data = pd.read_excel(file, usecols="y:z", sheet_name=depth)
rcht_data = pd.read_excel(file, usecols="aa:ab", sheet_name=depth)
rcht_data.dropna(how='all', inplace=True)
#varian_data = pd.read_excel(file, usecols="k:l", sheet_name=depth)
#varian_data = pd.read_excel(file, usecols="m:n", sheet_name=depth)
#varian_data = pd.read_excel(file, usecols="o:p", sheet_name=depth)
varian_data = pd.read_excel(file, usecols="q:R", sheet_name=depth)
varian_data.dropna(how='all', inplace=True)

field = '40x40'

rcht_results = ["CROSSPLANE_PROFILE", {'Type': 'CROSSPLANE_PROFILE', 'Modality': 'X', 'Energy': '10', 'Filter': 'FFF',
                                       'Nominal Field Size': field,
                        'Profile_pos': rcht_data['Depth' + field + '.1'].to_numpy(), 'Profile_val': rcht_data[field + '.1'].to_numpy()}]
varian_results = ["CROSSPLANE_PROFILE", {'Type': 'CROSSPLANE_PROFILE', 'Modality': 'X', 'Energy': '10', 'Filter': 'FFF',
                                         'Nominal Field Size': field,
                          'Profile_pos': varian_data['Depth' + field].to_numpy(), 'Profile_val': varian_data[field].to_numpy()}]

gamma_result = {'CROSSPLANE_PROFILE': Gamma(varian_results, rcht_results, lower_percent_dose_cutoff=20,
                                            dose_percent_threshold=3, distance_mm_threshold=1)}

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
gamma_curve(gamma_result['CROSSPLANE_PROFILE'], 'CROSSPLANE_PROFILE', ax[0])
gamma_hist(gamma_result['CROSSPLANE_PROFILE'], 'CROSSPLANE_PROFILE', ax[1])
fig.tight_layout()
plt.show()
pass
"""
### PDDs ####

file = r'X:\Physics\RPU\RayStation\Commissioning\BeamData\LA4\FinalSet_SmoothedOnceAndSymmetrised\10FFF\Golden Data Comparison\PDDs\PDDs.xlsx'


rcht_data = pd.read_excel(file, usecols="v:aa", nrows=299)
varian_data = pd.read_excel(file, usecols="A:F", nrows=301)

field = '40x40'

rcht_results = ["PDD", {'Type': 'PDD', 'Modality': 'X', 'Energy': '10.0', 'Filter': 'FFF', 'Nominal Field Size': field,
                        'PDD_pos': rcht_data['Depth' + '.2'].to_numpy(), 'PDD_val': rcht_data[field + '.2'].to_numpy()}]
varian_results = ["PDD", {'Type': 'PDD', 'Modality': 'X', 'Energy': '10.0', 'Filter': 'FFF', 'Nominal Field Size': field,
                          'PDD_pos': varian_data['Depth'].to_numpy(), 'PDD_val': varian_data[field].to_numpy()}]

gamma_result = {'PDD': Gamma(varian_results, rcht_results, lower_percent_dose_cutoff=20, dose_percent_threshold=3,
                distance_mm_threshold=1)}
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
gamma_curve(gamma_result['PDD'], 'PDD', ax[0])
gamma_hist(gamma_result['PDD'], 'PDD', ax[1])
fig.tight_layout()
plt.show()
pass

"""
