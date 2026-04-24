"""StockIQ — stock analysis package."""

__version__ = "2.0.0"

# Eagerly seed sys.modules with our subpackages. This reduces the window
# for Streamlit's hot-reload race on Python 3.13 where a subpackage like
# 'stockiq.core' gets dropped from sys.modules mid-rerun. Wrap in try so
# a transient failure here does not poison the whole package; normal
# lazy import will still work from the caller.
try:
    from . import core as core  # noqa: F401
    from . import data as data  # noqa: F401
    from . import models as models  # noqa: F401
    from . import ui as ui  # noqa: F401
except Exception:
    pass
