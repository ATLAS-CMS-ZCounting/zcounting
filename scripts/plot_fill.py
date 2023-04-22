import pandas as pd
import numpy as np
import json
import os,sys
import pdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

sys.path.append(os.getcwd())
from common import parsing, logging, utils, plotting

parser = parsing.parser()
parser.add_argument("-f", "--fills", default=[], type=int, nargs="*",
                    help="Fills to be plotted")
parser.add_argument("--no-ratio", action="store_true",
                    help="Make no ratio")
parser.add_argument("--fmts", default=["png", ], type=int, nargs="+",
                    help="List of formats to store the plots")
parser.add_argument("--no-ref", default=["png", ], action="store_true",
                    help="Don't show reference lumi")                    
args = parser.parse_args()
log = logging.setup_logger(__file__, args.verbose)

colors, textsize, labelsize, markersize = plotting.set_matplotlib_style()

color_cms = colors[0]
color_atlas = colors[1]
marker_cms = "o"
marker_atlas = "o"


with open(args.atlas_csv, "r") as ifile:
    df_atlas = pd.read_csv(ifile)

with open(args.cms_csv, "r") as ifile:
    df_cms = pd.read_csv(ifile)

# figure out which fills to plot
if args.fills != []:
    fills = args.fills
else:
    fills = set(np.concatenate([df_atlas["fill"].values, df_cms["fill"].values]))

if not os.path.isdir(args.outputDir):
    os.mkdir(args.outputDir)

def convert_time(df, atlas=False):
    # convert time
    df['timeDown'] = df['beginTime'].apply(lambda x: utils.to_datetime(x, atlas))
    df['timeUp'] = df['endTime'].apply(lambda x: utils.to_datetime(x, atlas))

    # center of each time slice
    df['timewindow'] = df['timeUp'] - df['timeDown']
    df['time'] = df['timeDown'] + df['timewindow']/2

    df["time"] = df["time"].apply(lambda x: utils.to_mpl_time(x, atlas))
    df["timeUp"] = df["timeUp"].apply(lambda x: utils.to_mpl_time(x, atlas))
    df["timeDown"] = df["timeDown"].apply(lambda x: utils.to_mpl_time(x, atlas))

convert_time(df_atlas, atlas=True) 
convert_time(df_cms) 


for fill in fills:
    if fill not in df_cms["fill"].values:
        log.info(f"Fill {fill} not found for CMS")
        continue
    if fill not in df_atlas["fill"].values:
        log.info(f"Fill {fill} not found for ATLAS")
        continue

    dfill_cms = df_cms.loc[df_cms["fill"] == fill]
    dfill_atlas = df_atlas.loc[df_atlas["fill"] == fill]

    if len(dfill_cms) == 1 and len(dfill_atlas):
        log.info(f"Only one measurement in fill {fill}, next fill!")
        continue

    log.info(f"Plot fill {fill}")

    def get_x(df):
        x = df['time'].values
        xUp = df['timeUp'].values
        xDown = df['timeDown'].values

        # convert into hours
        xUp = (xUp - x) * 24 
        xDown = (x - xDown) * 24
        x = x * 24
        x = (x - x[0] + xDown[0])

        return x, xUp, xDown
    
    x_cms, xUp_cms, xDown_cms = get_x(dfill_cms)
    x_atlas, xUp_atlas, xDown_atlas = get_x(dfill_atlas)

    # x axis range
    time_cms = max(x_cms) - min(x_cms)
    time_atlas = max(x_atlas) - min(x_atlas)

    xMin = min(min(x_cms-xDown_cms), min(x_atlas-xDown_atlas))
    xMax = max(max(x_cms+xUp_cms), max(x_atlas+xUp_atlas))
    xRange = xMax - xMin
    xMin = xMin - xRange * 0.015
    xMax = xMax + xRange * 0.015
    xRange = xMax - xMin

    def get_y(df):
        y = df['ZRate'].values

        # statistical error, for simplicity just take sqrt from delivered
        yErr = y * 1./np.sqrt(df['delZCount'].values)

        return y, yErr

    y_cms, yErr_cms = get_y(dfill_cms)
    y_atlas, yErr_atlas = get_y(dfill_atlas)

    y_lumi_cms = dfill_cms["instDelLumi"] * 650.
    y_lumi_atlas = dfill_atlas["instDelLumi"] * 650.

    ticksteps = 1 + xRange // 8 

    xTicks = np.arange(0, int(xMax), ticksteps)
    
    # make plot of Z boson rate as function of LHC fill time
    plt.clf()
    fig = plt.figure()
    if not args.no_ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
    else:
        ax1 = fig.add_subplot(111)
        
    fig.subplots_adjust(hspace=0.0, left=0.15, right=0.95, top=0.92, bottom=0.125)
        
    ax1.set_title(f"Fill {fill}")
    ax1.set_ylabel("Z boson rate [Hz]")
    
    # ax1.errorbar(x_cms, y_cms, xerr=(xDown_cms, xUp_cms), yerr=yErr_cms, label="CMS", color=color_cms, marker=marker_cms, mfc='none',
    #     linestyle='', zorder=0)

    # ax1.errorbar(x_atlas, y_atlas, xerr=(xDown_atlas, xUp_atlas), yerr=yErr_atlas, label="ATLAS", color=color_atlas, marker=marker_atlas,
    #     linestyle='', zorder=0)

    if not args.no_ref:
        ax1.errorbar(x_cms, y_lumi_cms, xerr=(xDown_cms, xUp_cms), label="CMS L", color=color_cms, 
            linestyle='', zorder=0)

        ax1.errorbar(x_atlas, y_lumi_atlas, xerr=(xDown_atlas, xUp_atlas), label="ATLAS L", color=color_atlas, 
            linestyle='', zorder=0)


    ax1.plot(x_cms, y_cms,label="CMS Z", color=color_cms, marker=marker_cms, mfc='none',
        linestyle='', zorder=0)

    ax1.plot(x_atlas, y_atlas, label="ATLAS Z", color=color_atlas, marker=marker_atlas, mfc='none',
        linestyle='', zorder=0)


    leg = ax1.legend(loc="upper right", ncol=2)

    yMin = min(min(y_cms-yErr_cms),min(y_atlas-yErr_atlas))
    yMax = max(max(y_cms+yErr_cms),max(y_atlas+yErr_atlas))

    yRange = yMax - yMin 
    ax1.set_ylim([yMin - yRange*0.15, yMax + yRange*0.15])
    ax1.set_xlim([xMin, xMax])
    ax1.set_xticks(xTicks)
    
    if not args.no_ratio:
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        ax2.set_xlabel("LHC runtime")
        ax2.set_ylabel("ATLAS / CMS")

        nPoints = 1000
        xx = np.linspace(xMin, xMax, nPoints)
        yy_atlas = np.array([y_atlas[x_atlas < x][-1] if any(x_atlas < x) else 0 for x in xx])
        yy_cms = np.array([y_cms[x_cms < x][-1] if any(x_cms < x) else 0 for x in xx])

        ratio = np.nan_to_num(yy_atlas/yy_cms, nan=1, posinf=1, neginf=1)

        if not args.no_ref:
            ax2.errorbar(x_cms, y_cms/y_lumi_cms, xerr=(xDown_cms, xUp_cms), label="Z", color=color_cms, 
                linestyle='', zorder=0)

            ax2.errorbar(x_atlas, y_atlas/y_lumi_atlas, xerr=(xDown_atlas, xUp_atlas), label="A", color=color_atlas, 
                linestyle='', zorder=0)

        # ax2.plot(xx, ratio, color="black", marker=None, linestyle='--', zorder=0)       
        
        ratio_cumsum = np.nan_to_num(np.cumsum(yy_atlas)/np.cumsum(yy_cms), nan=1, posinf=1, neginf=1)

        ax2.plot(xx, ratio_cumsum, color="black", marker=None,
            linestyle='-', zorder=0)

        ratio = round(ratio_cumsum[-1],3)
        ax2.text(0.4, 0.6 if ratio<1 else 0.2, "Integrated Z ratio: "+str(ratio), verticalalignment='bottom', transform=ax2.transAxes)

        ax2.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black",linestyle="--", linewidth=1)
        
        ax2.set_ylim([0.71,1.29])
        ax2.set_xlim([xMin, xMax])
        ax2.set_xticks(xTicks)

        # align y labels
        ax1.yaxis.set_label_coords(-0.12, 0.5)
        ax2.yaxis.set_label_coords(-0.12, 0.5)
    else:
        ax1.set_xlabel("LHC runtime")

    for fmt in args.fmts:
        plt.savefig(args.outputDir+f"/fill_{fill}.{fmt}")
    plt.close()

    # make plot of cumulative Z boson rate as function of LHC fill time
    def get_y(df):
        y = df["delZCount"].cumsum().values
        yErr = y * 1./np.sqrt(df["delZCount"].cumsum().values)

        return y, yErr

    y_cms, yErr_cms = get_y(dfill_cms)
    y_atlas, yErr_atlas = get_y(dfill_atlas)

    y_lumi_cms = dfill_cms["delLumi"].cumsum().values* 650.
    y_lumi_atlas = dfill_atlas["delLumi"].cumsum().values * 650.

    plt.clf()
    fig = plt.figure()
    if not args.no_ratio and not args.no_ref:
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
    elif not args.no_ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
    else:
        ax1 = fig.add_subplot(111)
        
    fig.subplots_adjust(hspace=0.0, left=0.15, right=0.95, top=0.92, bottom=0.125)
        
    ax1.set_title(f"Fill {fill}")
    ax1.set_ylabel("Number of Z bosons")
    
    # ax1.errorbar(x_cms, y_cms, xerr=(xDown_cms, xUp_cms), yerr=yErr_cms, label="CMS", color=color_cms, marker=marker_cms,
    #     linestyle='', zorder=0)

    # ax1.errorbar(x_atlas, y_atlas, xerr=(xDown_atlas, xUp_atlas), yerr=yErr_atlas, label="ATLAS", color=color_atlas, marker=marker_atlas,
    #     linestyle='', zorder=0)

    if not args.no_ref:
        ax1.errorbar(x_cms, y_lumi_cms, xerr=(xDown_cms, xUp_cms), label="CMS L", color=color_cms, 
            linestyle='', zorder=0)

        ax1.errorbar(x_atlas, y_lumi_atlas, xerr=(xDown_atlas, xUp_atlas), label="ATLAS L", color=color_atlas, 
            linestyle='', zorder=0)

    ax1.plot(x_cms, y_cms,label="CMS Z", color=color_cms, marker=marker_cms, mfc='none',
        linestyle='', zorder=0)

    ax1.plot(x_atlas, y_atlas, label="ATLAS Z", color=color_atlas, marker=marker_atlas, mfc='none',
        linestyle='', zorder=0)

    leg = ax1.legend(loc="lower right", ncol=2)

    yMin = min(min(y_cms-yErr_cms),min(y_atlas-yErr_atlas))
    yMax = max(max(y_cms+yErr_cms),max(y_atlas+yErr_atlas))

    yRange = yMax - yMin 
    ax1.set_ylim([yMin - yRange*0.15, yMax + yRange*0.15])
    ax1.set_xlim([xMin, xMax])
    ax1.set_xticks(xTicks)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    
    if not args.no_ratio:
        ax1.xaxis.set_major_locator(ticker.NullLocator())

        # cumulative Z ratio
        ax2.set_ylabel("A / C")

        nPoints = 1000
        xx = np.linspace(xMin, xMax, nPoints)
        yy_atlas = np.array([y_atlas[x_atlas < x][-1] if any(x_atlas < x) else 1 for x in xx])
        yy_cms = np.array([y_cms[x_cms < x][-1] if any(x_cms < x) else 1 for x in xx])

        ax2.plot(xx, yy_atlas/yy_cms, color="black", marker=None,
            linestyle='-', zorder=0)

        ratio = round(y_atlas[-1] / y_cms[-1],3)

        ax2.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black", linestyle="--", linewidth=1)
        
        ax2.text(0.4, 0.6 if ratio<1 else 0.2, "Integrated Z ratio: "+str(ratio), verticalalignment='bottom', transform=ax2.transAxes)

        ax2.set_ylim([0.81,1.19])
        ax2.set_xlim([xMin, xMax])
        ax2.set_xticks(xTicks)

        if not args.no_ref:
            ax2.xaxis.set_major_locator(ticker.NullLocator())
    
            # cumulative lumi ratio
            ax3.set_xlabel("LHC runtime")
            ax3.set_ylabel("A / C")

            yy_lumi_atlas = np.array([y_lumi_atlas[x_atlas < x][-1] if any(x_atlas < x) else 1 for x in xx])
            yy_lumi_cms = np.array([y_lumi_cms[x_cms < x][-1] if any(x_cms < x) else 1 for x in xx])

            ax3.plot(xx, yy_lumi_atlas/yy_lumi_cms, color="black", marker=None,
                linestyle='--', zorder=0)

            ratio = round(y_lumi_atlas[-1] / y_lumi_cms[-1],3)

            ax3.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black", linestyle="--", linewidth=1)
            
            ax3.text(0.4, 0.6 if ratio<1 else 0.2, "Integrated L ratio: "+str(ratio), verticalalignment='bottom', transform=ax3.transAxes)

            ax3.set_ylim([0.81,1.19])
            ax3.set_xlim([xMin, xMax])
            ax3.set_xticks(xTicks)

            ax3.yaxis.set_label_coords(-0.12, 0.5)

        # align y labels
        ax1.yaxis.set_label_coords(-0.12, 0.5)
        ax2.yaxis.set_label_coords(-0.12, 0.5)

    else:
        ax1.set_xlabel("LHC runtime")

    for fmt in args.fmts:
        plt.savefig(args.outputDir+f"/fill_cumulative_{fill}.{fmt}")
    plt.close()
    


