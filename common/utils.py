import matplotlib as mpl
from datetime import datetime
import pandas as pd
import uncertainties as unc
from common.logging import child_logger
import pdb

log = child_logger(__name__)

def load_csv_files(filenames):

    dfs = []
    for filename in filenames:
        with open(filename, "r") as ifile:
            dfs.append(pd.read_csv(ifile))
    df = pd.concat(dfs)

    for col in ["ZRate", "delZCount"]:
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

    zeros = (df["ZRate"] <= 0) & (df["delZCount"] <= 0) & (df["delLumi"] <= 0) & (df["instDelLumi"] <= 0)
    nZeros = sum(zeros)

    if nZeros > 0:
        log.info(f"Found {nZeros} empty measurements, those will be removed.")
    
    return df[~zeros]

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
