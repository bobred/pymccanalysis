import matplotlib.pyplot as plt
import os
import pandas as pd
from pymccanalysis.wtscans import WTScanPlotting
from pymccanalysis.gamma_analysis import gamma_hist, gamma_curve
from pymccanalysis.readmcc import ReadMCC


base_dir = os.path.dirname(__file__)
file_dir = os.path.join(base_dir, 'samples')

ptw_scan = ReadMCC(file_dir + r'\X06 OPEN 10X10 PDDCRIN WAT 210220 all.mcc')
ptw_scan.process_data()
ptw_ref = ReadMCC(file_dir + r'\ref_data\X06 OPEN 10X10 PDDCRIN.mcc')

ptw_ref.process_data()

ptw_scan_results = ptw_scan.mcc_data
ptw_ref_results = ptw_ref.mcc_data

# only select those that are present in the reference data
dd = list(ptw_scan_results.keys() & ptw_ref_results.keys())
gg = dict(sorted({k: ptw_ref_results[k] for k in dd if k in ptw_ref_results}.items(),  reverse=True))

gamma = ReadMCC.calc_gamma(ptw_ref_results, ptw_scan_results)
scans = WTScanPlotting()
pass

fig, ax = plt.subplots(3, figsize=(10, 16))
scans.pdd_plot(ptw_scan_results, ax[0])
scans.profile_plot(ptw_scan_results['INPLANE_PROFILE_100'], 'INPLANE_PROFILE', ax[1])
scans.profile_plot(ptw_scan_results['CROSSPLANE_PROFILE_100'], 'CROSSPLANE_PROFILE', ax[2])
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(3, 2, figsize=(18, 16))
gamma_curve(gamma['PDD'], 'PDD', ax[0, 0])
gamma_hist(gamma['PDD'], 'PDD', ax[0, 1])
gamma_curve(gamma['INPLANE_PROFILE_100'], 'INPLANE_PROFILE', ax[1, 0])
gamma_hist(gamma['INPLANE_PROFILE_100'], 'INPLANE_PROFILE', ax[1, 1])
gamma_curve(gamma['CROSSPLANE_PROFILE_100'], 'CROSSPLANE_PROFILE', ax[2, 0])
gamma_hist(gamma['CROSSPLANE_PROFILE_100'], 'CROSSPLANE_PROFILE', ax[2, 1])
fig.tight_layout()
plt.show()

pass
"""
def get_table(**kwargs):
    _keys = [key for key, value in kwargs.items()]
    _values = [value for key, value in kwargs.items()]

    ref = _values[0]
    if ref['Type'] == 'PDD':
        ref.pop('PDD_pos', None)
        ref.pop('PDD_val', None)
    elif 'PROFILE' in ref['Type']:
        ref.pop('Profile_pos', None)
        ref.pop('Profile_val', None)
    ref_values = [value for key, value in ref.items()]
    ref_keys = [key for key, value in ref.items()]
    scan = _values[1]
    if scan['Type'] == 'PDD':
        scan.pop('PDD_pos', None)
        scan.pop('PDD_val', None)
    elif 'PROFILE' in scan['Type']:
        scan.pop('Profile_pos', None)
        scan.pop('Profile_val', None)
    scan_values = [value for key, value in scan.items()]
    df = pd.DataFrame({'': ref_keys, 'Reference': ref_values, 'Scan': scan_values})

    #df.set_index(df.columns[1])
    
    dd = defaultdict(list)
    for d in (pdds['ref'], pdds['scan']):  # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)
    #df = pd.DataFrame(pdd.items())
    return df

table_pdd = get_table(ref = ptw_ref_results['PDD'], scan = ptw_scan_results['PDD'])
table_cross = get_table(ref = ptw_ref_results['CROSSPLANE_PROFILE_100'], scan = ptw_scan_results['CROSSPLANE_PROFILE_100'])
table_in = get_table(ref = ptw_ref_results['INPLANE_PROFILE_100'], scan = ptw_scan_results['INPLANE_PROFILE_100'])
writer = pd.ExcelWriter('test.xlsx',engine='xlsxwriter')
workbook=writer.book
worksheet=workbook.add_worksheet('Validation')
writer.sheets['Validation'] = worksheet
table_pdd.to_excel(writer, sheet_name='Validation', startrow=0, startcol=0, index=False)
table_cross.to_excel(writer,sheet_name='Validation',startrow=16, startcol=0, index=False)
table_in.to_excel(writer,sheet_name='Validation',startrow=34, startcol=0, index=False)
workbook.close()"""
pass