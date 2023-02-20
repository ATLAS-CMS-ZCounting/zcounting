import pandas as pd
import numpy as np
import json
import os,sys
import pdb

sys.path.append(os.getcwd())
from common import parsing, logging

parser = parsing.parser()
args = parser.parse_args()
log = logging.setup_logger(__file__, args.verbose)

with open(args.atlas_csv, "r") as ifile:
    df_atlas = pd.read_csv(ifile)

with open(args.cms_csv, "r") as ifile:
    df_cms = pd.read_csv(ifile)

for fill in set(np.concatenate([df_atlas["fill"].values, df_cms["fill"].values])):
    if fill not in df_cms["fill"].values:
        log.info(f"Fill {fill} not found for CMS")
        continue
    if fill not in df_atlas["fill"].values:
        log.info(f"Fill {fill} not found for ATLAS")
        continue

    log.info(f"Now at fill {fill}")

    pdb.set_trace()

    dfill_cms = df_cms.loc[df_cms["fill"] == fill]
    dfill_atlas = df_atlas.loc[df_atlas["fill"] == fill]

    # TODO figure out how this can actually be done with different time intervals

    


