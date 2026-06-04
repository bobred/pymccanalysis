import os
from pymccanalysis.readmcc import ReadMCC
import matplotlib.pyplot as plt
from pymccanalysis.wtscans import WTScanPlotting
from pymccanalysis.gamma_analysis import gamma_hist, gamma_curve


base_dir = os.path.dirname(__file__)
file_dir = os.path.join(base_dir, 'samples')

ptw_scan = ReadMCC(file_dir + r"\E06 15x15 PDD.mcc")
ptw_scan.process_data()
ptw_ref = ReadMCC(file_dir + r'\ref_data\Depth dose curves 15x15.mcc', energy=6)
ptw_ref.process_data()

ptw_scan_results = ptw_scan.mcc_data
ptw_ref_results = ptw_ref.mcc_data

gamma = ReadMCC.calc_gamma(ptw_ref_results, ptw_scan_results)
scans = WTScanPlotting()

fig, ax = plt.subplots(1, figsize=(8, 8))
scans.pdd_plot(ptw_scan_results, ax)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20, 12))
gamma_curve(gamma['PDD'], 'PDD', ax[0])
gamma_hist(gamma['PDD'], 'PDD', ax[1])
fig.tight_layout()
plt.show()
pass