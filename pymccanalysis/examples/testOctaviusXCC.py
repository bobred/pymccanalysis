from pymccanalysis.readmcc import ReadMCC
import matplotlib.pyplot as plt
import matplotlib as mpl
from pymccanalysis.detector_array import ArrayScanPlotting
from pylinac import FieldProfileAnalysis, Centering, Edge, Normalization, Interpolation
from pylinac.core.profile import FWXMProfile
from pylinac.core.image import load

array = ReadMCC(r"X:\Physics\RPU\Octavius4D\Comissioning\Gantry QA\LA3 post ion chamber replacement gantry rotation"
                r"\X06 6X_G0_DR600 NONE 25X25 ~ 0 BEA 230127 19'36 4D.xcc", profile_depths=[5], array_profiles=True, scale=1)

img = load(array.mcc_data['PROFILE_GRID'])

fa = FieldProfileAnalysis(img)
fa.analyze(normalization=Normalization.BEAM_CENTER, edge_type=Edge.INFLECTION_DERIVATIVE, invert=True)

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