import pandas as pd
import numpy as np
import json
import os,sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import json

import pdb

sys.path.append(os.getcwd())
from common import parsing, logging, utils, plotting


parser = parsing.parser()
parser.add_argument("-b", "--beginFill", default=None, type=int, 
                    help="First fill to be considered")
parser.add_argument("-e", "--endFill", default=None, type=int, 
                    help="Last fill to be considered")
parser.add_argument("--no-ratio", action="store_true",
                    help="Make no ratio")
parser.add_argument("--fmts", default=["png", ], type=int, nargs="+",
                    help="List of formats to store the plots")
parser.add_argument("--rrange", default=[0.81, 1.19], type=float, nargs=2,
                    help="Y range of lower ratio plot")
parser.add_argument("--yrange", default=[0.6, 1.9], type=float, nargs=2,
                    help="Y range of upper ratio plot")
args = parser.parse_args()

log = logging.setup_logger(__file__, args.verbose)

colors, textsize, labelsize, markersize = plotting.set_matplotlib_style()

color_zbosons = "blue"
color_lumi = "red"

if not os.path.isdir(args.outputDir):
    os.mkdir(args.outputDir)

df_atlas = utils.load_csv_files(args.atlas_csv)
df_cms = utils.load_csv_files(args.cms_csv)

def select_fills(df):
    df_new = df
    if args.beginFill:
        df_new = df_new.loc[df_cms["fill"] <= args.beginFill]

    if args.endFill:
        df_new = df_new.loc[df_cms["fill"] >= args.endFill]

    return df_new

df_atlas = select_fills(df_atlas)
df_cms = select_fills(df_cms)

def convert_time(df):
    # convert time
    df['timeDown'] = df['beginTime'].apply(lambda x: utils.to_datetime(x))
    df['timeUp'] = df['endTime'].apply(lambda x: utils.to_datetime(x))

    # center of each time slice
    df['timewindow'] = df['timeUp'] - df['timeDown']
    df['timewindow'] = df['timewindow'].apply(lambda x: x.total_seconds())

convert_time(df_atlas)
convert_time(df_cms)

# compute integrated rate and lumi
df_atlas["intZRate"] = df_atlas['timewindow']*df_atlas["ZRate"]
df_cms["intZRate"] = df_cms['timewindow']*df_cms["ZRate"]

df_atlas["intDelLumi"] = df_atlas['timewindow']*df_atlas["instDelLumi"]
df_cms["intDelLumi"] = df_cms['timewindow']*df_cms["instDelLumi"]

# calculate integrated Z rate and lumi per fill
dfill_atlas = df_atlas.groupby("fill")[["delZCount","intZRate", "delLumi", "intDelLumi"]].sum()
dfill_cms = df_cms.groupby("fill")[["delZCount","intZRate", "delLumi", "intDelLumi"]].sum()

def rename(df, name):
    rename = {key: f"{name}_{key}" for key in df.keys() if key != "fill"}
    return df.rename(columns=rename)

dfill_atlas = rename(dfill_atlas, "atlas")
dfill_cms = rename(dfill_cms, "cms")

dfill = pd.concat([dfill_atlas, dfill_cms], axis=1, join="inner") 

# compute Z boson count and luminosity ratios between ATLAS and CMS
dfill["ratio_NZ"] = dfill["atlas_delZCount"] / dfill["cms_delZCount"]
dfill["ratio_intNZ"] = dfill["atlas_intZRate"] / dfill["cms_intZRate"]

dfill["ratio_Lumi"] = dfill["atlas_delLumi"] / dfill["cms_delLumi"] / 1000.
dfill["ratio_intLumi"] = dfill["atlas_intDelLumi"] / dfill["cms_intDelLumi"] / 1000.

# total sums
ratio_NZ = dfill["atlas_delZCount"].sum() / dfill["cms_delZCount"].sum()
ratio_intNZ = dfill["atlas_intZRate"].sum() / dfill["cms_intZRate"].sum()

ratio_Lumi = dfill["atlas_delLumi"].sum() / dfill["cms_delLumi"].sum()
ratio_intLumi = dfill["atlas_intDelLumi"].sum() / dfill["cms_intDelLumi"].sum() / 1000.

log.info(f"Ratio NZ: {ratio_NZ}")
log.info(f"Ratio int NZ: {ratio_intNZ}")
log.info(f"Ratio L: {ratio_Lumi}")
log.info(f"Ratio int L: {ratio_intLumi}")

# statistical uncertainty - sinple sqrt(N)
dfill["ratio_err_NZ"] = dfill["ratio_NZ"] * ( 1/dfill["atlas_delZCount"] + 1/dfill["cms_delZCount"] )**0.5

# systematic uncertainty
dfill["ratio_err_NZ"] = (dfill["ratio_err_NZ"]**2 + (np.ones(len(dfill))*0.04)**2)**0.5

fills = dfill.index

# --- json file for LPC
dout = dfill[["ratio_intNZ", "ratio_err_NZ"]]
dout = dout.rename(columns={"ratio_intNZ":"rat", "ratio_err_NZ": "err"})

dout["fillno"] = dout.index.values.astype(str)
result = json.loads(dout.to_json(orient="index", index=True))
with open(args.outputDir+"/rate_ratio.json", "w") as ofile:
    json.dump(result, ofile, indent=4)

# --- Make plot of ratios as a function of the fills
minFill = min(fills)
maxFill = max(fills)

xMin = minFill
xMax = maxFill
xRange = xMax - xMin
xMin = xMin - xRange * 0.015
xMax = xMax + xRange * 0.015
xRange = xMax - xMin

# y_lumi = dfill["ratio_Lumi"]
# y_z = dfill["ratio_NZ"]
y_lumi = dfill["ratio_intLumi"]
y_z = dfill["ratio_intNZ"]
y_ratio = y_z/y_lumi

yerr_z = dfill["ratio_err_NZ"]
yerr_lumi = np.ones(len(fills))*0.03
yerr_ratio = np.sqrt(yerr_z**2 + yerr_lumi**2)

plt.clf()
fig = plt.figure(figsize=(10.0,4.0))
if not args.no_ratio:
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
else:
    ax1 = fig.add_subplot(111)

fig.subplots_adjust(hspace=0.0, left=0.075, right=0.97, top=0.99, bottom=0.125)
    
ax1.set_ylabel("ATLAS / CMS")
ax1.set_xlabel("LHC fill number")

ax1.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black",linestyle="--", linewidth=1)

ax1.errorbar(fills, y_z, yerr=yerr_z, label="Z bosons", color=color_zbosons, 
            linestyle='', marker='o', mfc='none' , zorder=1)

ax1.errorbar(fills, y_lumi, yerr=yerr_lumi, label="Lumi", color=color_lumi, 
            linestyle='', marker='o', mfc='none' , zorder=1)

ax1.text(0.01, 0.95, "{\\bf{ATLAS+CMS}} "+"\\emph{"+args.label+"} \n", verticalalignment='top', horizontalalignment="left", transform=ax1.transAxes)

leg = ax1.legend(loc="upper right", ncol=2)

ax1.set_ylim(args.yrange)
ax1.set_xlim([xMin, xMax])

if not args.no_ratio:

    ax2.set_ylabel(r"$R_\mathrm{Z} / R_\mathrm{\mathcal{L}}$")
    ax2.set_xlabel("LHC fill number")

    ax2.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black",linestyle="--", linewidth=1)

    ax2.errorbar(fills, y_ratio, yerr=yerr_ratio, label="Z bosons", color="black", 
            linestyle='', marker='.', mfc='none' , zorder=1)

    ax2.set_ylim(args.rrange)
    ax2.set_xlim([xMin, xMax])

for fmt in args.fmts:
    plt.savefig(args.outputDir+f"/ratio_fill_{minFill}_to_{maxFill}.{fmt}")
plt.close()
