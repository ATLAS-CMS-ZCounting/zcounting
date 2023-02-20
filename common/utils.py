import matplotlib as mpl
from datetime import datetime

def to_mpl_time(timestring):
    # our agreed format has "YY/MM/DD HH:MM:SS"
    datetime_object = datetime.strptime(timestring, '%y/%m/%d %H:%M:%S')
   
    return mpl.dates.date2num(datetime_object)
