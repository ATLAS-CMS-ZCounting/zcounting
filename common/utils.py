import matplotlib as mpl
from datetime import datetime
import pandas as pd
import uncertainties as unc
from common.logging import child_logger
import numpy as np
import pdb

log = child_logger(__name__)

def load_csv_files(filenames, fills=None, threshold_outlier=0, xsec=700):

    dfs = []
    for filename in filenames:
        with open(filename, "r") as ifile:
            dfs.append(pd.read_csv(ifile))
    df = pd.concat(dfs, ignore_index=True)

    df = df.rename(columns={"ZRate": "delZRate"})

    for col in ["delZRate", "delZCount"]:
        if col in df.keys():
            if df[col].dtype==object:
                df[col] = df[col].apply(lambda x: unc.ufloat_fromstr(x).n)
        else:
            log.error(f"Column '{col}' not found in input file but is expected.")

    if "delLumi" in df.keys():        
        # convert into /pb (should be O(10) )
        if max(df["delLumi"]) > 1000:
            log.warning(f"Automatic conversion of 'delLumi' into /pb")
            df["delLumi"] /= 1000
    else:
        log.error(f"Column 'delLumi' not found in input file but is expected.")

    if "instDelLumi" in df.keys():        
        # convert into /pb (should be O(0.01) )
        if max(df["instDelLumi"]) > 1.0:
            log.warning(f"Automatic conversion of 'instDelLumi' into /pb")
            df["instDelLumi"] /= 1000
    else:
        log.error(f"Column 'instDelLumi' not found in input file but is expected.")

    if fills is not None and fills != []:
        log.debug(f"Select fills {fills}")
        df = df.loc[df["fill"].apply(lambda x, fs=fills: x in fs)]

    mask = (df["delZRate"] <= 0) & (df["delZCount"] <= 0) & (df["delLumi"] <= 0) & (df["instDelLumi"] <= 0)
    nZeros = sum(mask)

    if nZeros > 0:
        log.info(f"Found {nZeros} empty measurements, those will be removed.")

    nans = np.isnan(df["delZRate"]) | np.isnan(df["delZCount"]) | np.isnan(df["delLumi"]) | np.isnan(df["instDelLumi"])
    nNans = sum(nans)

    if nNans > 0:
        log.info(f"Found {nNans} measurements with NaN, those will be removed.")

    mask = mask | nans

    if threshold_outlier > 0:
        pulls = df["delZRate"].values / (df["instDelLumi"].values * xsec) - 1

        outliers = abs(pulls) > threshold_outlier
        nOutliers = sum(outliers)        

        if nOutliers > 0:
            log.info(f"Found {nOutliers} outlier measurements, those will be removed.")
        
        mask = mask | outliers

    return df[~mask]

def to_mpl_time(timestring):
    datetime_object = to_datetime(timestring)
   
    return mpl.dates.date2num(datetime_object)

def to_datetime(timestring):
    
    # our agreed format has "YY/MM/DD HH:MM:SS", but need to catch some derivations

    if isinstance(timestring, str):
        if timestring[0] == ' ':
            timestring = timestring[1:]
        
        if int(timestring[:2]) <= 12:
            formatstr = '%m/%d/%y %H:%M:%S'
        else:
            formatstr = '%y/%m/%d %H:%M:%S'

        datetime_object = datetime.strptime(timestring, formatstr)
    else:
        datetime_object = timestring
    
    return datetime_object

def overlap(df1, df2):
    # only keep measurements that overlap (at least partially)
    log.debug("Remove overlap")

    filtered_rows = []

    # loop over all fills
    for fill, g1 in df1.groupby("fill"):            
        g2 = df2.loc[df2["fill"] == fill]

        # Iterate over each row in df2
        for _, row in g1.iterrows():
            # Extract the "beginTime" and "endTime" of the current row in df2
            begin_time = row['beginTime']
            end_time = row['endTime']
            
            # Check if any row in df1 has its "beginTime" and "endTime" outside the current row's interval
            if any([(begin_time <= e) and (end_time >= b) for b,e in g2[["beginTime","endTime"]].values] ):
                # Add the current row from df2 to the list of filtered rows
                filtered_rows.append(row)

    log.debug("Done removing overlap")

    # Create a new DataFrame from the filtered rows
    return pd.DataFrame(filtered_rows)