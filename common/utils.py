import matplotlib as mpl
from datetime import datetime

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