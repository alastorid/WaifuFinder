#!/usr/bin/env python3
"""Check all24 SDPA input/mask tensor ranks before publishing a release."""
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
import coremltools as ct
from wd_tagger_attention import check_sdpa_spec
package = Path(sys.argv[1])
spec = ct.utils.load_spec(str(package / 'Data/com.apple.CoreML/model.mlmodel'))
print(json.dumps({'nativeSDPABlocks': check_sdpa_spec(spec), 'allQueryKeyValueMaskRanks': 4}))
