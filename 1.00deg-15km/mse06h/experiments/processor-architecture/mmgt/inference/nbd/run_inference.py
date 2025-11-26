import os
import sys
_nested_eagle = os.path.expandvars("${HOME}/nested-eagle")
sys.path.append(_nested_eagle)

import eagle.inference

if __name__ == "__main__":
    eagle.inference.run_inference()
