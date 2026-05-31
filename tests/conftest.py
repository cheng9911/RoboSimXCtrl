"""
conftest.py -- pytest configuration for RoboSimXCtrl tests.

Handles the circular import in robot/__init__.py by pre-loading robot
submodules before the package __init__ tries to import ur5e.
"""

import sys
import os
import importlib
import importlib.util
import types

# ---------------------------------------------------------------------------
# Path setup -- MUST happen before any `import robot`
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROBOT_DIR = os.path.join(PROJECT_ROOT, "robot")

# Ensure bare imports work (e.g. `from robot_config import RobotConfig`)
for p in [ROBOT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Pre-load robot submodules to break the circular import
# ---------------------------------------------------------------------------

def _load_module_from_file(module_name, file_path):
    """Load a Python module directly from its file path."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# 1. Load robot.robot (the base Robot class module)
robot_robot = _load_module_from_file(
    "robot.robot", os.path.join(ROBOT_DIR, "robot.py"))

# 2. Create a stub package so `from robot import X` works
if "robot" not in sys.modules:
    pkg = types.ModuleType("robot")
    pkg.__path__ = [ROBOT_DIR]
    pkg.__package__ = "robot"
    pkg.__file__ = os.path.join(ROBOT_DIR, "__init__.py")
    sys.modules["robot"] = pkg
else:
    pkg = sys.modules["robot"]

# Expose key symbols on the package
for attr in ("Robot", "get_transformation_mdh", "wrap"):
    setattr(pkg, attr, getattr(robot_robot, attr))

# 3. Load robot_config (needed by ur5e.py)
_load_module_from_file("robot_config", os.path.join(ROBOT_DIR, "robot_config.py"))

# 4. Load robot.ur5e, robot.iiwa14, robot.diana
_CLASS_MAP = {
    "ur5e": "UR5e",
    "iiwa14": "IIWA14",
    "diana": "Diana",
}
for mod_name, file_name in [
    ("robot.ur5e", "ur5e.py"),
    ("robot.iiwa14", "iiwa14.py"),
    ("robot.diana", "diana.py"),
]:
    fpath = os.path.join(ROBOT_DIR, file_name)
    if os.path.exists(fpath):
        try:
            sub = _load_module_from_file(mod_name, fpath)
            short_name = mod_name.split(".")[-1]
            cls_name = _CLASS_MAP.get(short_name)
            if cls_name and hasattr(sub, cls_name):
                setattr(pkg, cls_name, getattr(sub, cls_name))
        except Exception:
            pass  # Diana may fail if tracikpy is not installed

# 5. Now make the __init__.py content available (it was already loaded indirectly)
# Patch the package __init__ to have the expected exports
if not hasattr(pkg, "__all__"):
    pkg.__all__ = ["Robot", "UR5e", "IIWA14", "Diana"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import pytest
import numpy as np


@pytest.fixture
def ur5e_robot():
    """Return an instantiated UR5e robot."""
    from robot import UR5e
    return UR5e()


@pytest.fixture
def iiwa14_robot():
    """Return an instantiated IIWA14 robot."""
    from robot import IIWA14
    return IIWA14()
