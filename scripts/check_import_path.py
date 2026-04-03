import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(_ROOT, "Pointcept"))
sys.path.insert(0, os.path.join(_ROOT, "Pointcept"))
import pointcept.models  # noqa: F401

with open(os.path.join(_ROOT, "import_check.txt"), "w", encoding="utf-8") as f:
    f.write(pointcept.models.__file__ + "\n")
