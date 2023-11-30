# Z boson counting
This repository collects scripts that we use to process the Z boson counting data in a collabortative manner between the ATLAS and CMS experiments.

## Scripts
This folder contains scripts to produce comparison plots and make computations.

### scripts/plot_fill.py
This script generates for each fill two plots as a function of the LHC filltime. 
The first plot shows the Z boson rate extracted by ATLAS and CMS. 
A ratio is included by default that shows the cumulative ratio of the Z boson rates.
The reference luminosity (scaled with a single constant) can be shown as comparison with '--ref-lumi'.

An example to run the script:
```
python3 scripts/plot_fill.py -a path/to/atlas*.csv -c path/to/cms.csv --ref-lumi -o output/folder
```

### scripts/compute_zrate_ratio.py
This script generates a .json file that is given to LPC to make Z boson rate and luminosity comparisons between ATLAS and CMS. 
The LPC wants the following format:
```
{ "6961": {"rat": 0.9863303068675684,
           "fillno": "6961",
           "err": 0.04184645090855196},
  "6960": {"rat": 0.9940399833569503,
           "fillno": "6960",
           "err": 0.04217354478013575},
  ...
} 
```
It also produces a plot with the Z boson rate and luminosity ratios between ATLAS and CMS and the double ratio. 

An example to run the script:
```
python3 scripts/compute_zrate_ratio.py -a path/to/atlas*.csv -c path/to/cms.csv -o output/folder --label Preliminary
```

## common
This folder contains helper functions used in other places of the framework

# Additional information

## Agreed conventions

### phase space
We measure $\mathrm{Z}/\gamma^{\star}->\mu\mu$ events where intermediate $\tau$ leptons are excluded. 
The fiducial phase space, defined for bare muons, is:
- Both muons must have $p_{\mathrm{T}} > 27$ GeV
- Both muons must have $|\eta| < 2.4$ GeV
- The invariant mass of the dimuon system must be within $66 < m_{\mu\mu} < 116$ GeV.

### csv files
ATLAS and CMS share information with the Z bosons rate measurements about every 20 minutes. 
The measured number of Z bosons is corrected for the efficiency and corrected to bare muons in the fiducial phase space as defined above. 
For each experiment, only the good quality data is analyzed. Thus, the intervalls usually don't agree in start and end time, and a perfect overlap is not given.
The information is shared in a .csv file with the following columns:
```
fill,beginTime,endTime,recZRate,instRecLumi,delLumi,delZCount
```
- fill: The LHC fill number
- beginTime: The begin time of the measurement as a string in the format "YY/MM/DD HH:MM:SS" in UTC.
- endTime: The same as beginTime but for the end of the measurement. 
- instRecLumi: The instantaneous recorded luminosity during the measurement (in $\mathrm{pb}^{-1}\mathrm{s}^{-1}$). 
- recZRate: The number of measured Z bosons divided by the time of the measurement in seconds (This can be seen as the recorded Z boson rate, which is independent on the reference luminosity). 
- delLumi: The integrated delivered luminosity measured during the measurement (in $\mathrm{pb}^{-1}$). 
- delZCount: The number of Z bosons divided by the recorded luminosity during the measurement multiplied by the delivered luminosity in the total time of the measurement (This can be seen as the delivered number of Z bosons during the total time of the measuremend. The reference luminosity is used to extrapolate to the total number of produced Z bosons, independent of the detector). 

## LPC
- Run 2: https://lpc.web.cern.ch/plots.html?year=2018&runtype=protons
(Go to the tab 'ATLAS/CMS Z-counting ratio')

## ATLAS-CMS meetings
- 2022/06/27: https://indico.cern.ch/event/1175147/
- 2023/02/13: https://indico.cern.ch/event/1253518/
- 2023/06/30: https://indico.cern.ch/event/1300368/
