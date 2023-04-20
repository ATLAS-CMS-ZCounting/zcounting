import matplotlib as mpl
from datetime import datetime

def to_mpl_time(timestring, atlas=False):
    # our agreed format has "YY/MM/DD HH:MM:SS"
    if atlas:
        formatstr = ' %y/%m/%d %H:%M:%S'
    else:
        formatstr = '%y/%m/%d %H:%M:%S'

    if isinstance(timestring, str):
        datetime_object = datetime.strptime(timestring, formatstr)
    else:
        datetime_object = timestring
   
    return mpl.dates.date2num(datetime_object)
