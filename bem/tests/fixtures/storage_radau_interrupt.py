"""Test-only abrupt parent exits after handoff rename or partial IFFT write."""

import argparse
import os
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--stop-stage", choices=["handoff", "weights"], required=True)
args, remaining = parser.parse_known_args()
tool = Path(__file__).resolve().parents[2] / "tools" / "prepare_storage_radau.py"
module = runpy.run_path(str(tool))
namespace = module["main"].__globals__
if args.stop_stage == "handoff":
    original = namespace["HANDOFF"]["handoff_raw"]

    def interrupted(*values, **keywords):
        def stop(event):
            if event["phase"] == "renamed" and event["branch"] == 0:
                os._exit(81)

        return original(*values, **keywords, progress=stop)

    namespace["HANDOFF"]["handoff_raw"] = interrupted
else:
    direct = namespace["FINISH"]["finish"].__globals__["DIRECT"]
    original = direct["invert_physical_fortran"]

    def interrupted(*values, **keywords):
        def stop(event):
            if (
                event["phase"] == "column_written"
                and event["family"] == 1
                and event["first"] == 0
                and event["column"] == 0
            ):
                os._exit(81)

        return original(*values, **keywords, progress=stop)

    direct["invert_physical_fortran"] = interrupted
sys.argv = [str(tool), *remaining]
module["main"]()
