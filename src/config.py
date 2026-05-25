"""Load and expose runtime-configurable thresholds."""
import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.yaml")
_DEFAULTS = {"flag": 15.0, "warning": 10.0}


def load_asym_thresholds() -> dict:
    """Return {'flag': float, 'warning': float} from thresholds.yaml, with fallbacks."""
    try:
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        asym = data.get("asymmetry", {})
        return {
            "flag":    float(asym.get("flag",    _DEFAULTS["flag"])),
            "warning": float(asym.get("warning", _DEFAULTS["warning"])),
        }
    except Exception:
        return dict(_DEFAULTS)
