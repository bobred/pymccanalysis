"""from readmcc import ReadMCC
import matplotlib.pyplot as plt
from wtscans import WTScanPlotting
from gamma_analysis import gamma_hist, gamma_curve
"""
import os
import matplotlib.pyplot as plt
from pymccanalysis.wtscans import WTScanPlotting
from pymccanalysis.gamma_analysis import gamma_hist, gamma_curve
from pymccanalysis.readmcc import ReadMCC


base_dir = os.path.dirname(__file__)
file_dir = os.path.join(base_dir, 'samples')

ptw_scan = ReadMCC(file_dir + r'\X10 FFF OPEN 10X10 PDDCRIN WAT 210220.mcc')
ptw_scan.process_data()
ptw_ref = ReadMCC(file_dir + r'\ref_data\X10FFF OPEN 10X10 PDDCRIN.mcc')
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