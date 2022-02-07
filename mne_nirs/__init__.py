from ._version import __version__

from . import channels
from . import datasets
from . import experimental_design
from . import io
from . import preprocessing
from . import signal_enhancement
from . import simulation
from . import statistics
from . import utils
from . import visualisation
from . import visualisation as viz  # for MNE-Python users

__all__ = ['__version__']
