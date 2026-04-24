"""StockIQ — stock analysis package."""

__version__ = "2.0.0"

# Eagerly register subpackages in sys.modules. Streamlit's script-reload
# path on Python 3.13 has been observed to drop 'stockiq.core' (and peers)
# from sys.modules between reruns while keeping 'stockiq' itself, which
# causes KeyError at importlib._find_and_load_unlocked when the next rerun
# does `from stockiq.core.analyzer import ...`.
from . import core as core  # noqa: F401
from . import data as data  # noqa: F401
from . import models as models  # noqa: F401
from . import ui as ui  # noqa: F401
