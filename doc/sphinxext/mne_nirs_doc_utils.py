"""Doc-build helpers that sphinx-gallery imports by name.

With a parallel gallery build the examples run in worker processes that never
execute ``conf.py``, so anything they need (scrapers, per-example setup) has to
live in an importable module like this one rather than in ``conf.py`` itself.
"""

import warnings

import mne


def reset_modules(gallery_conf, fname):
    """Set up the 3D backend before each example (also in parallel workers)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import pyvista

    mne.viz.set_3d_backend("pyvistaqt")
    pyvista.OFF_SCREEN = False
    pyvista.BUILDING_GALLERY = True


report_scraper = mne.report._ReportScraper()
gui_scraper = mne.gui._GUIScraper()
brain_scraper = mne.viz._brain._BrainScraper()
mne_qt_browser_scraper = mne.viz._scraper._MNEQtBrowserScraper()
