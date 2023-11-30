import pandas as pd
import numpy as np
import json
import os,sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.dates as md

import pdb

sys.path.append(os.getcwd())
from common import parsing, logging, utils, plotting

parser = parsing.parser()
parser.add_argument("-f", "--fills", default=[], type=int, nargs="*",
                    help="Fills to be plotted")
parser.add_argument("--no-ratio", action="store_true",
                    help="Make no ratio")
parser.add_argument("--ref-lumi", default=False, action="store_true",
                    help="Show reference lumi")
args = parser.parse_args()
log = logging.setup_logger(__file__, args.verbose, not args.noColorLogger)

colors, textsize, labelsize, markersize = plotting.set_matplotlib_style()

color_cms = colors[0]
color_atlas = colors[1]
marker_cms = "o"
marker_atlas = "o"

label_ratio = r"$\frac{\mathrm{ATLAS}}{\mathrm{CMS}}$"
label_ratio_ref = r"$\frac{N_\mathrm{Z}}{\mathcal{L}}$"

log.info("Load ATLAS csv file")
df_atlas = utils.load_csv_files(args.atlas_csv, args.fills, threshold_outlier=args.threshold_outlier, scale=args.scale_atlas)
log.info("Load CMS csv file")
df_cms = utils.load_csv_files(args.cms_csv, args.fills, threshold_outlier=args.threshold_outlier, scale=args.scale_cms)

# figure out which fills to plot
if args.fills != []:
    fills = args.fills
else:
    fills = set(np.concatenate([df_atlas["fill"].values, df_cms["fill"].values]))

if not os.path.isdir(args.outputDir):
    os.mkdir(args.outputDir)

def convert_time(df):
    # convert time
    df['beginTime'] = df['beginTime'].apply(lambda x: utils.to_datetime(x))
    df['endTime'] = df['endTime'].apply(lambda x: utils.to_datetime(x))

    # center of each time slice
    df['timewindow'] = df['endTime'] - df['beginTime']
    df['time'] = df['beginTime'] + df['timewindow']/2

    df['timewindow'] = df['timewindow'].apply(lambda x: x.total_seconds())

    df["time"] = df["time"].apply(lambda x: utils.to_mpl_time(x))
    df["timeUp"] = df["endTime"].apply(lambda x: utils.to_mpl_time(x))
    df["timeDown"] = df["beginTime"].apply(lambda x: utils.to_mpl_time(x))

    df["timeUp"] = df["timeUp"] - df["time"]
    df["timeDown"] = df["time"] - df["timeDown"]

convert_time(df_atlas) 
convert_time(df_cms) 

if args.overlap:
    # only keep measurements that overlap at least partially
    df_cms, df_atlas = utils.overlap(df_cms, df_atlas), utils.overlap(df_atlas, df_cms) 

for fill in fills:
    if fill not in df_cms["fill"].values:
        log.info(f"Fill {fill} not found for CMS")
        continue
    if fill not in df_atlas["fill"].values:
        log.info(f"Fill {fill} not found for ATLAS")
        continue

    dfill_cms = df_cms.loc[df_cms["fill"] == fill]
    dfill_atlas = df_atlas.loc[df_atlas["fill"] == fill]

    # compute ratio of full fill
    def sum_rate(df):
        return sum(df['timewindow']*df["delZRate"])

    nz_rate_atlas = sum_rate(dfill_atlas)
    nz_rate_cms = sum_rate(dfill_cms)
    nz_rate_ratio = nz_rate_atlas / nz_rate_cms

    log.debug(f"NZ(ATLAS): {nz_rate_atlas}")
    log.debug(f"NZ(CMS): {nz_rate_cms}")
    log.debug(f"NZ(ATLAS)/NZ(CMS): {nz_rate_ratio}")

    log.debug(f"-----------")

    nz_atlas = sum(dfill_atlas["delZCount"])
    nz_cms = sum(dfill_cms["delZCount"])
    nz_ratio = nz_atlas / nz_cms

    log.debug(f"NZ(ATLAS): {nz_atlas}")
    log.debug(f"NZ(CMS): {nz_cms}")
    log.debug(f"NZ(ATLAS)/NZ(CMS): {nz_ratio}")


    if len(dfill_cms) <= 1 or len(dfill_atlas) <= 1:
        log.info(f"Only one measurement in fill {fill} in both experiments, next fill!")
        continue

    log.info(f"Plot fill {fill}")

    def get_x(df):
        x = df['time'].values
        xUp = df['timeUp'].values
        xDown = df['timeDown'].values

        return x, xUp, xDown
    
    x_cms, xUp_cms, xDown_cms = get_x(dfill_cms)
    x_atlas, xUp_atlas, xDown_atlas = get_x(dfill_atlas)

    log.debug("Set x-axis")

    # x axis range
    xMin = min(min(x_cms-xDown_cms), min(x_atlas-xDown_atlas))
    xMax = max(max(x_cms+xUp_cms), max(x_atlas+xUp_atlas))
    xRange = xMax - xMin
    xMin = xMin - xRange * 0.015
    xMax = xMax + xRange * 0.015
    xRange = xMax - xMin

    log.debug("Get total timewindow")

    # total timewindow in seconds
    dateMin = mpl.dates.num2date(xMin)
    dateMax = mpl.dates.num2date(xMax)

    if dateMax.month != dateMin.month:
        datestring = "{0}/{1} - {2}/{3}/{4}".format(dateMin.month, dateMin.day, dateMax.month, dateMax.day, dateMin.year)
    elif dateMax.day != dateMin.day:
        datestring = "{0}-{1}/{2}/{3}".format(dateMin.day, dateMax.day, dateMin.month, dateMin.year)
    else:
        datestring = "{0}/{1}/{2}".format(dateMin.day, dateMin.month, dateMin.year)

    timewindow = (dateMax - dateMin).total_seconds()

    # fine grid for plotting smooth rate
    nPoints = 10000
    xGrid = np.linspace(xMin, xMax, nPoints)

    # time delta
    dt = timewindow/nPoints

    def set_xaxis_format(axis, time=True):
        axis.set_xlim([xMin, xMax])
        axis.set_xlabel("LHC runtime [h]")
        if time:
            # locator = md.AutoDateLocator(minticks=2, maxticks=24)
            # formatter = md.ConciseDateFormatter(locator)
            # axis.xaxis.set_major_locator(locator)
            # axis.xaxis.set_major_formatter(formatter)

            xfmt = md.DateFormatter('%H:00')
            axis.xaxis.set_major_formatter(xfmt)
        else:
            ticksteps = 1 + xRange // 8 
            xTicks = np.arange(0, int(xMax), ticksteps)
            axis.set_xticks(xTicks)


    # ---  make plot with delZRate from ATLAS and CMS
    log.debug("Make plot with delZRate for ATLAS and CMS")

    def get_y(df):
        y = df['delZRate'].values

        # statistical error, for simplicity just take sqrt from delivered
        yErr = y * 1./np.sqrt(df['delZCount'].values)
        np.nan_to_num(yErr, copy=False)

        return y, yErr

    y_cms, yErr_cms = get_y(dfill_cms)
    y_atlas, yErr_atlas = get_y(dfill_atlas)

    y_lumi_cms = dfill_cms["instDelLumi"].values * dfill_cms["xsec_theory"]
    y_lumi_atlas = dfill_atlas["instDelLumi"].values * dfill_atlas["xsec_theory"]

    # make plot of Z boson rate as function of LHC fill time
    plt.clf()
    fig = plt.figure()
    if not args.no_ratio and args.ref_lumi:
        fig = plt.figure(figsize=(6.0,6.0))
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
        
    ax1.set_ylabel("Z boson rate [Hz]")

    ax1.text(0.0, 0.99, f"Fill {fill}", verticalalignment='bottom', transform=ax1.transAxes)    
    ax1.text(1.0, 0.99, datestring, verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes)

    ax1.plot(x_cms, y_cms, label="CMS", color=color_cms, marker=marker_cms, mfc='none',
        linestyle='', zorder=2)

    ax1.plot(x_atlas, y_atlas, label="ATLAS", color=color_atlas, marker=marker_atlas, mfc='none',
        linestyle='', zorder=2)

    ax1.text(0.03, 0.95, "{\\bf{ATLAS+CMS}} "+"\\emph{"+args.label+"} \n", verticalalignment='top', horizontalalignment="left", transform=ax1.transAxes)

    if args.ref_lumi:
        # plot reference lumi scaled to Z rate
        ax1.errorbar(x_cms, y_lumi_cms, xerr=(xDown_cms, xUp_cms), label="CMS L", color=color_cms, 
            linestyle='', zorder=1)

        ax1.errorbar(x_atlas, y_lumi_atlas, xerr=(xDown_atlas, xUp_atlas), label="ATLAS L", color=color_atlas, 
            linestyle='', zorder=1)

    leg = ax1.legend(loc="upper right", ncol=2)

    yMin = min(min(y_cms-yErr_cms),min(y_atlas-yErr_atlas))
    yMax = max(max(y_cms+yErr_cms),max(y_atlas+yErr_atlas))

    yRange = yMax - yMin 
    ax1.set_ylim([yMin - yRange*0.01, yMax + yRange*0.3])

    set_xaxis_format(ax1)

    if not args.no_ratio:
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        ax2.set_ylabel(label_ratio)

        ax2.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black",linestyle="--", linewidth=1)

        # for each point we want the rate
        def get_rate(point_x, x, xUp, xDown, y):
            rate = y[(point_x < x + xUp) & (point_x > x - xDown)]
            if len(rate) == 0:
                return 0
            elif len(rate)==1: 
                return rate[0]
            else:
                raise RuntimeError("Multiple rates found at given point!")

        yy_rate_atlas = np.array([get_rate(x, x_atlas, xUp_atlas, xDown_atlas, y_atlas) for x in xGrid])
        yy_rate_cms = np.array([get_rate(x, x_cms, xUp_cms, xDown_cms, y_cms) for x in xGrid])

        yy_atlas = np.cumsum(yy_rate_atlas) * dt
        yy_cms = np.cumsum(yy_rate_cms) * dt

        log.debug(f"From delZRate")
        log.debug(f"Total number of Z (ATLAS): {yy_atlas[-1]}")
        log.debug(f"Total number of Z (CMS): {yy_cms[-1]}")

        # only show ratio if both, ATLAS and CMS != 0
        indices = (yy_atlas!=0) & (yy_cms!=0)

        xx_ratio = xGrid[indices]
        yy_ratio = yy_atlas[indices] / yy_cms[indices]

        ax2.plot(xx_ratio, yy_ratio, color="black", marker=None, linestyle='-', zorder=0)

        intZ_a = round(yy_atlas[-1])
        intZ_c = round(yy_cms[-1])
        ratio_end = round(yy_atlas[-1] / yy_cms[-1], 3)

        ax2.text(0.4, 0.75 if ratio_end<1 else 0.25, "$R_\mathrm{Z}$ = "+f"{intZ_a}/{intZ_c} = {ratio_end}", verticalalignment='bottom', transform=ax2.transAxes)

        ax2.set_ylim([0.71,1.29])
        set_xaxis_format(ax2)

        if args.ref_lumi:
            log.debug("Add reference lumi in plots")
            ax2.xaxis.set_major_locator(ticker.NullLocator())

            # cumulative sum of lumi
            log.debug("Make cumulative sum of lumi")
            import pdb
            pdb.set_trace()
            yy_atlas_lumi = np.cumsum([get_rate(x, x_atlas, xUp_atlas, xDown_atlas, y_lumi_atlas) for x in xGrid]) * dt
            yy_cms_lumi = np.cumsum([get_rate(x, x_cms, xUp_cms, xDown_cms, y_lumi_cms) for x in xGrid]) * dt

            xx_ratio_lumi = xGrid[indices]
            yy_ratio_lumi = yy_atlas_lumi[indices] / yy_cms_lumi[indices]

            ax2.plot(xx_ratio_lumi, yy_ratio_lumi, color="black", marker=None, linestyle='--', zorder=0)

            intL_a = round(yy_atlas_lumi[-1])
            intL_c = round(yy_cms_lumi[-1])
            ratio_end_lumi = round(yy_atlas_lumi[-1] / yy_cms_lumi[-1], 3)
            ax2.text(0.4, 0.55 if ratio_end_lumi<1 else 0.05, "$R_\mathrm{\mathcal{L}}$ = "+f"{intL_a}/{intL_c} = {ratio_end_lumi}", verticalalignment='bottom', transform=ax2.transAxes)

            # cumulative lumi ratio
            log.debug("Make cumulative lumi ratio")
            ax3.set_ylabel(label_ratio_ref)

            ax3.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black",linestyle="--", linewidth=1)

            ax3.errorbar(x_cms, y_cms/y_lumi_cms, xerr=(xDown_cms, xUp_cms), label="Z", color=color_cms, 
                linestyle='', zorder=0)

            ax3.errorbar(x_atlas, y_atlas/y_lumi_atlas, xerr=(xDown_atlas, xUp_atlas), label="A", color=color_atlas, 
                linestyle='', zorder=0)

            ax3.set_ylim([0.81,1.19])

            set_xaxis_format(ax3)

            ax3.yaxis.set_label_coords(-0.12, 0.5)

        # align y labels
        ax1.yaxis.set_label_coords(-0.12, 0.5)
        ax2.yaxis.set_label_coords(-0.12, 0.5)

    
    for fmt in args.fmts:
        outname = f"{args.outputDir}/fill_{fill}.{fmt}"
        log.info(f"Save figure: {outname}")
        plt.savefig(outname)
    plt.close()


    # --- make plot of cumulative Z boson rate as function of LHC fill time
    log.info(f"Plot fill {fill} with cumulative numbers")
    def get_y(df):
        y = df["delZCount"].cumsum().values
        yErr = np.sqrt(df["delZCount"].cumsum().values)

        return y, yErr

    y_cms, yErr_cms = get_y(dfill_cms)
    y_atlas, yErr_atlas = get_y(dfill_atlas)

    y_lumi_cms = dfill_cms["delLumi"].cumsum().values * dfill_cms["xsec_theory"]
    y_lumi_atlas = dfill_atlas["delLumi"].cumsum().values * dfill_atlas["xsec_theory"]

    plt.clf()
    fig = plt.figure()
    if not args.no_ratio and args.ref_lumi:
        fig = plt.figure(figsize=(6.0,6.0))
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
        
    ax1.set_ylabel("Number of Z bosons")
    ax1.text(0.1, 0.99, f"Fill {fill}", verticalalignment='bottom', transform=ax1.transAxes)    
    ax1.text(1.0, 0.99, datestring, verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes)

    if args.ref_lumi:
        ax1.errorbar(x_cms, y_lumi_cms, xerr=(xDown_cms, xUp_cms), label="CMS L", color=color_cms, 
            linestyle='', zorder=1)

        ax1.errorbar(x_atlas, y_lumi_atlas, xerr=(xDown_atlas, xUp_atlas), label="ATLAS L", color=color_atlas, 
            linestyle='', zorder=1)

    ax1.plot(x_cms, y_cms,label="CMS Z", color=color_cms, marker=marker_cms, mfc='none',
        linestyle='', zorder=2)

    ax1.plot(x_atlas, y_atlas, label="ATLAS Z", color=color_atlas, marker=marker_atlas, mfc='none',
        linestyle='', zorder=2)

    ax1.text(0.03, 0.95, "{\\bf{ATLAS+CMS}} "+"\\emph{"+args.label+"} \n", verticalalignment='top', horizontalalignment="left", transform=ax1.transAxes)

    leg = ax1.legend(loc="lower right", ncol=2)

    yMin = min(min(y_cms-yErr_cms),min(y_atlas-yErr_atlas))
    yMax = max(max(y_cms+yErr_cms),max(y_atlas+yErr_atlas))

    yRange = yMax - yMin 
    ax1.set_ylim([yMin - yRange*0.01, yMax + yRange*0.3])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(5,5))

    set_xaxis_format(ax1)

    if not args.no_ratio:
        ax1.xaxis.set_major_locator(ticker.NullLocator())

        # cumulative Z ratio
        ax2.set_ylabel(label_ratio)

        yy_atlas = np.array([y_atlas[x_atlas < x][-1] if any(x_atlas < x) else 1 for x in xGrid])
        yy_cms = np.array([y_cms[x_cms < x][-1] if any(x_cms < x) else 1 for x in xGrid])

        ax2.plot(xGrid, yy_atlas/yy_cms, color="black", marker=None,
            linestyle='-', zorder=0)

        intL_a = round(y_atlas[-1])
        intL_c = round(y_cms[-1])
        ratio = round(y_atlas[-1] / y_cms[-1],3)

        ax2.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black", linestyle="--", linewidth=1)
        
        ax2.text(0.4, 0.6 if ratio<1 else 0.2, "$R_\mathrm{Z}$ = "+f"{intL_a}/{intL_c} = {ratio}", verticalalignment='bottom', transform=ax2.transAxes)

        ax2.set_ylim([0.81,1.19])
        set_xaxis_format(ax2)

        log.debug(f"From delZCount")
        log.debug(f"Total number of Z (ATLAS): {y_atlas[-1]}")
        log.debug(f"Total number of Z (CMS): {y_cms[-1]}")

        if args.ref_lumi:
            ax2.xaxis.set_major_locator(ticker.NullLocator())
    
            # cumulative lumi ratio
            ax3.set_ylabel(label_ratio)

            yy_lumi_atlas = np.array([y_lumi_atlas[x_atlas < x][-1] if any(x_atlas < x) else 1 for x in xGrid])
            yy_lumi_cms = np.array([y_lumi_cms[x_cms < x][-1] if any(x_cms < x) else 1 for x in xGrid])

            ax3.plot(xGrid, yy_lumi_atlas/yy_lumi_cms, color="black", marker=None,
                linestyle='--', zorder=0)

            intL_a = round(y_lumi_atlas[-1])
            intL_c = round(y_lumi_cms[-1])
            ratio = round(y_lumi_atlas[-1] / y_lumi_cms[-1],3)

            ax3.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black", linestyle="--", linewidth=1)
            
            ax3.text(0.4, 0.6 if ratio<1 else 0.2, "$R_\mathrm{\mathcal{L}}$ = "+f"{intL_a}/{intL_c} = {ratio}", verticalalignment='bottom', transform=ax3.transAxes)

            ax3.set_ylim([0.81,1.19])
            set_xaxis_format(ax3)

            ax3.yaxis.set_label_coords(-0.12, 0.5)

        # align y labels
        ax1.yaxis.set_label_coords(-0.12, 0.5)
        ax2.yaxis.set_label_coords(-0.12, 0.5)


    for fmt in args.fmts:
        outname = f"{args.outputDir}/fill_cumulative_{fill}.{fmt}"
        log.info(f"Save figure: {outname}")
        plt.savefig(outname)

    plt.close()
    


