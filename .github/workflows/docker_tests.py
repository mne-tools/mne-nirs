import mne
import numpy as np

size = (600, 600)
renderer = mne.viz.backends.renderer.create_3d_figure(
    bgcolor="w", size=size, scene=False
)
mne.viz.set_3d_backend("pyvista")
print("Creating image")
renderer.sphere((0, 0, 0), "k", 1, resolution=1000)
renderer.plotter.camera.enable_parallel_projection()
renderer.figure.plotter.camera.SetParallelScale(1)
renderer.show()
data = (renderer.screenshot() / 255.0).mean(-1)  # colors
renderer.close()
print("Validating image")
want = np.ones(size)
dists = np.sqrt(
    np.linspace(-1, 1, size[0])[:, np.newaxis] ** 2
    + np.linspace(-1, 1, size[1]) ** 2
)
want = (dists > 0.5).astype(float)
corr = np.corrcoef(want.ravel(), data.ravel())[0, 1]
assert 0.99 <= corr <= 1
print("Tests passed!")
