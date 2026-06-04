from readgolden import ReadGoldenData
from os import path
from pymccanalysis.readmcc import ReadMCC
import matplotlib.pyplot as plt
from pymccanalysis.wtscans import WTScanPlotting
from pymccanalysis.gamma_analysis import gamma_hist, gamma_curve


base_dir = path.dirname(__file__)
file_dir = path.join(base_dir, 'samples')

ptw_scan = ReadMCC(file_dir + r'\X10 FFF OPEN 10X10 PDDCRIN WAT 210220.mcc')
ptw_scan.process_data()
# gd = ReadGoldenData(r'D:\MEGA\VARIAN Golden Beam data', 'Truebeam', '6MV')
ref = ReadGoldenData(r'C:\Users\Murphyja\Documents\TrueBeam Representative Beam Data for Eclipse', 'Truebeam', '10FFF')

ptw_scan_results = ptw_scan.mcc_data
ptw_ref_results = ref.mcc_data
# only select those that are present in the reference data

gamma = ReadMCC.calc_gamma(ptw_ref_results, ptw_scan_results)
scans = WTScanPlotting()

fig, ax = plt.subplots(2, figsize=(10, 12))
scans.pdd_plot(ptw_scan_results, ax[0])
scans.profile_plot(ptw_scan_results['CROSSPLANE_PROFILE_100'], 'CROSSPLANE_PROFILE', ax[1])
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(18, 16))
gamma_curve(gamma['PDD'], 'PDD', ax[0, 0])
gamma_hist(gamma['PDD'], 'PDD', ax[0, 1])
gamma_curve(gamma['CROSSPLANE_PROFILE_100'], 'CROSSPLANE_PROFILE', ax[1, 0])
gamma_hist(gamma['CROSSPLANE_PROFILE_100'], 'CROSSPLANE_PROFILE', ax[1, 1])
fig.tight_layout()
plt.show()

pass

