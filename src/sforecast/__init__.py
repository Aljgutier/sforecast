# read version from installed package
from importlib.metadata import version
__version__ = version("sforecast")

from .sforecast import sliding_forecast
from .sforecast import sforecast