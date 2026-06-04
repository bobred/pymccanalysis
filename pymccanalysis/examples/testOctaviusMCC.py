import os
from pymccanalysis.readmcc import ReadMCC
import matplotlib.pyplot as plt
from pymccanalysis.detector_array import ArrayScanPlotting, DetectorArray
import matplotlib as mpl


base_dir = os.path.dirname(__file__)
file_dir = os.path.join(base_dir, 'samples')

array = ReadMCC(file_dir + r"\X06 6X_G0_DR600 NONE 25X25 ~ 0 BEA 230127 19'36 4D.mcc", profile_depths=[5],
                array_profiles=True)
array.process_data()
arr_plot = ArrayScanPlotting()

fig = plt.figure()
gspec = mpl.gridspec.GridSpec(ncols=1, nrows=1)

ax1 = fig.add_subplot(gspec[0, 0])

ax1.imshow(array.mcc_data['PROFILE_GRID'], extent=(-130, 130, -130, 130))
plt.show()

fig4, ax = plt.subplots(2, figsize=(8, 12))
arr_plot.profile_plot(array.mcc_data, 'INPLANE_PROFILE', ax[0])
arr_plot.profile_plot(array.mcc_data, 'CROSSPLANE_PROFILE', ax[1])
fig4.tight_layout()
plt.show()
pass
