# Z boson counting
This repository collects scripts that we use to process the Z boson counting data in a collabortative manner between the ATLAS and CMS experiments.

## Scripts
This folder contains scripts to produce comparison plots and make computations
### scripts/plot_fill.py
This script generates for each fill two plots as a function of the LHC filltime. 
The first plot shows the Z boson rate extracted by ATLAS and CMS. 
A ratio is included by default that shows the cumulative ratio of the Z boson rates.
The reference luminosity (scaled with a single constant) can be shown as comparison with '--ref-lumi'.
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
ATLAS and CMS share information with several measurements of the Z boson rate (about every 20 minutes) in a .csv file with the following columns:
```
fill,beginTime,endTime,ZRate,instDelLumi,delLumi,delZCount
```
- fill: The LHC fill number
- beginTime: The begin time of the measurement as a string in the format "YY/MM/DD HH:MM:SS" in UTC.
- endTime: The same as beginTime but for the end of the measurement. 
- ZRate: The efficiency-corrected number of Z bosons, corrected to bare muons in the fiducial phase space of the measurement, divided by the active time of the measurement in seconds.   
- instDelLumi: The integrated luminosity measured during the measurement, divided by the time of the measurement in seconds ($\mathrm{pb}^{-1}\mathrm{s}^{-1}$). 
- delLumi: The integrated luminosity measured during the measurement ($\mathrm{pb}^{-1}$). 
- delZCount: The efficiency-corrected number of Z bosons, corrected to bare muons in the fiducial phase space of the measurement, divided by the active time of the measurement multiplied by the total time of the measurement. 

## LPC
- Run 2: https://lpc.web.cern.ch/plots.html?year=2018&runtype=protons
(Go to the tab 'ATLAS/CMS Z-counting ratio')

## ATLAS-CMS meetings
- 2022/06/27: https://indico.cern.ch/event/1175147/
- 2023/02/13: https://indico.cern.ch/event/1253518/
