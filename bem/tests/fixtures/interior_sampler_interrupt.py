"""Test-only abrupt exit after a rectangular raw/compact chunk is committed."""

import argparse
import os
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--stop-channel", required=True)
parser.add_argument("--after-count", type=int, default=8)
args, remaining = parser.parse_known_args()
path = Path(__file__).resolve().parents[2] / "tools/prepare_interior_spectra.py"
module = runpy.run_path(str(path))
namespace = module["main"].__globals__
original = namespace["CHECKPOINT"]["Writer"]


class InterruptedWriter(original):
    def flush(self):
        super().flush()
        if self.name == args.stop_channel and self.committed >= args.after_count:
            os._exit(83)


namespace["CHECKPOINT"]["Writer"] = InterruptedWriter
sys.argv = [str(path), *remaining]
module["main"]()
