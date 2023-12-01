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
parser.add_argument("--rrange", default=[0.81, 1.19], type=float, nargs=2,
                    help="Y range of lower ratio plot")
parser.add_argument("--yrange", default=[0.6, 1.9], type=float, nargs=2,
                    help="Y range of upper ratio plot")
args = parser.parse_args()

log = logging.setup_logger(__file__, args.verbose, not args.noColorLogger)

colors, textsize, labelsize, markersize = plotting.set_matplotlib_style()

color_zbosons = "blue"
color_lumi = "red"

if not os.path.isdir(args.outputDir):
    os.mkdir(args.outputDir)

df_atlas = utils.load_csv_files(args.atlas_csv, threshold_outlier=args.threshold_outlier, scale=args.scale_atlas)
df_cms = utils.load_csv_files(args.cms_csv, threshold_outlier=args.threshold_outlier, scale=args.scale_cms)

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
df_atlas["intdelZRate"] = df_atlas['timewindow']*df_atlas["delZRate"]
df_cms["intdelZRate"] = df_cms['timewindow']*df_cms["delZRate"]

df_atlas["intDelLumi"] = df_atlas['timewindow']*df_atlas["instDelLumi"]
df_cms["intDelLumi"] = df_cms['timewindow']*df_cms["instDelLumi"]

if args.overlap:
    # only keep measurements that overlap at least partially
    df_cms, df_atlas = utils.overlap(df_cms, df_atlas), utils.overlap(df_atlas, df_cms) 

# calculate integrated Z rate and lumi per fill
dfill_atlas = df_atlas.groupby("fill")[["delZCount","intdelZRate", "delLumi", "intDelLumi"]].sum()
dfill_cms = df_cms.groupby("fill")[["delZCount","intdelZRate", "delLumi", "intDelLumi"]].sum()

def rename(df, name):
    rename = {key: f"{name}_{key}" for key in df.keys() if key != "fill"}
    return df.rename(columns=rename)

dfill_atlas = rename(dfill_atlas, "atlas")
dfill_cms = rename(dfill_cms, "cms")

dfill = pd.concat([dfill_atlas, dfill_cms], axis=1, join="inner") 

fills = dfill.index

def zyield_ratio(df, zyield_atlas, zyield_cms, lumi_atlas, lumi_cms, postfix):
    log.info(f"Process z yield and luminosity ratios for {postfix}")

    # append at output file name
    if args.overlap:
        append = "_overlap"
    else:
        append = ""

    # print out total sums
    sum_z_atlas = round(df[zyield_atlas].sum(),1)
    sum_z_cms = round(df[zyield_cms].sum(),1)
    sum_l_atlas = round(df[lumi_atlas].sum()/1000,1)
    sum_l_cms = round(df[lumi_cms].sum()/1000,1)

    ratio_NZ = df[zyield_atlas].sum() / df[zyield_cms].sum()
    ratio_Lumi = df[lumi_atlas].sum() / df[lumi_cms].sum()
    ratio_double = ratio_NZ / ratio_Lumi

    log.info(f"Total ratio NZ: {ratio_NZ}")
    log.info(f"Total ratio L: {ratio_Lumi}")
    log.info(f"Total double ratio: {ratio_double}")

    # ratios per fill
    df["rat"] = df[zyield_atlas] / df[zyield_cms]
    df["rat_lumi"] = df[lumi_atlas] / df[lumi_cms]

    # statistical uncertainty
    df["err"] = df["rat"] * ( 1/df[zyield_atlas] + 1/df[zyield_cms] )**0.5
    # add ad-hoc systematic uncertainty
    systematic_uncertainty = (0.03**2+0.03**2)**0.5
    df["err"] = (df["err"]**2 + (np.ones(len(df))*systematic_uncertainty)**2)**0.5

    # --- json file for LPC
    dout = df[["rat", "err", "rat_lumi"]].copy()
    dout["fillno"] = dout.index.values.astype(str)
    result = json.loads(dout.to_json(orient="index", index=True))
    with open(args.outputDir+f"/zyield_ratio_{postfix}{append}.json", "w") as ofile:
        json.dump(result, ofile, indent=4)

    # --- Make plot of ratios as a function of the fills
    y_z = df["rat"]
    y_lumi = df["rat_lumi"]

    double_ratio = y_z/y_lumi

    minFill = min(fills)
    maxFill = max(fills)

    xMin = minFill
    xMax = maxFill
    xRange = xMax - xMin
    xMin = xMin - xRange * 0.015
    xMax = xMax + xRange * 0.015
    xRange = xMax - xMin

    yerr_z = df["err"]
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

    ax1.text(0.4, 0.95, "$R_\mathrm{\mathcal{L}}$ = "+ str(f"{sum_l_atlas}/{sum_l_cms} = {round(ratio_Lumi,3)}"), verticalalignment='top', transform=ax1.transAxes)
    ax1.text(0.4, 0.85, "$R_\mathrm{Z}$ = "+ str(f"{round(ratio_NZ,3)}"), verticalalignment='top', transform=ax1.transAxes)


    leg = ax1.legend(loc="upper right", ncol=2)

    ax1.set_ylim(args.yrange)
    ax1.set_xlim([xMin, xMax])

    if not args.no_ratio:
        # plot double ratio
        ax1.xaxis.set_major_locator(ticker.NullLocator())

        ax2.set_ylabel(r"$R_\mathrm{Z} / R_\mathrm{\mathcal{L}}$")
        ax2.set_xlabel("LHC fill number")

        ax2.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black",linestyle="--", linewidth=1)

        ax2.errorbar(fills, double_ratio, yerr=yerr_ratio, label="Z bosons", color="black", 
                linestyle='', marker='.', mfc='none' , zorder=1)

        ax2.set_ylim(args.rrange)
        ax2.set_xlim([xMin, xMax])

    for fmt in args.fmts:
        plt.savefig(args.outputDir+f"/ratio_fills_{minFill}_to_{maxFill}_{postfix}{append}.{fmt}")
    plt.close()


#zyield_ratio(dfill, "atlas_intdelZRate", "cms_intdelZRate", "atlas_intDelLumi", "cms_intDelLumi", "integrated")
zyield_ratio(dfill, "atlas_delZCount", "cms_delZCount", "atlas_delLumi", "cms_delLumi", "cumulated")
