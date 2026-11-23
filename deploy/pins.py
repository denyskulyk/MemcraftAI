"""Write deploy/pins.json from this PC, or check this PC against it.

    .venv\\Scripts\\python.exe deploy\\pins.py           # write the pins again
    .venv\\Scripts\\python.exe deploy\\pins.py --check   # list mismatches; exit 1 if any

The first run hashes every model file the two workflows load (minutes); later runs use
data/pins-cache.json.
"""

import argparse
import sys

from memcraft.pins import PINS_FILE, check_installation, write_pins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args()
    if args.check:
        problems = check_installation()
        for line in problems:
            print(line)
        print("pins match" if not problems else f"{len(problems)} mismatch(es)")
        return 1 if problems else 0
    pins = write_pins()
    print(
        f"wrote {PINS_FILE}: ComfyUI {pins['comfyui']['version']}, "
        f"{len(pins['custom_nodes'])} custom nodes, {len(pins['models'])} models"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
