from ._version import __version__

from . import signal_enhancement
from . import experimental_design

__all__ = ['__version__', 'create_boxcar', 'create_hrf']
