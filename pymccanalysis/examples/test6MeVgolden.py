from readgolden import ReadGoldenData
from os import path
from pymccanalysis.readmcc import ReadMCC
import matplotlib.pyplot as plt
from pymccanalysis.wtscans import WTScanPlotting
from pymccanalysis.gamma_analysis import gamma_hist, gamma_curve


base_dir = path.dirname(__file__)
file_dir = path.join(base_dir, 'samples')

ptw_scan = ReadMCC(file_dir + r'\Depth dose curves 15x15.mcc', energy=6)
ptw_scan.process_data()
# gd = ReadGoldenData(r'D:\MEGA\VARIAN Golden Beam data', 'Truebeam', '6MV')
ref = ReadGoldenData(r'C:\Users\Murphyja\Documents\TrueBeam Representative Beam Data for Eclipse',
                     'Truebeam', '6MeV', 15)

ptw_scan_results = ptw_scan.mcc_data
ptw_ref_results = ref.mcc_data
# only select those that are present in the reference data

gamma = ReadMCC.calc_gamma(ptw_ref_results, ptw_scan_results)
scans = WTScanPlotting()

fig, ax = plt.subplots(1, figsize=(10, 12))
scans.pdd_plot(ptw_scan_results, ax)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18, 12))
gamma_curve(gamma['PDD'], 'PDD', ax[0])
gamma_hist(gamma['PDD'], 'PDD', ax[1])
fig.tight_layout()
plt.show()

pass

