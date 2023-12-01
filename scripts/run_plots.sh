#!/bin/bash
echo "Awake"

# setings

ATLASBASE=/eos/home-c/cmszcont/data/zcounting/ATLAS/
ATLASSUB=2023_June_prel/
ATLAS=$ATLASBASE/$ATLASSUB

CMSBASE=/eos/home-c/cmszcont/data/zcounting/Run3/
CMSSUB=2023/
CMS=$CMSBASE/$CMSSUB

WEBBASE=/eos/home-c/cmszcont/www/public/ATLAS-CMS/Run3/
WEBSUB=2023/
WEBDIR=$WEBBASE/$WEBSUB

BASEDIR=/afs/cern.ch/user/c/cmszcont/zcounting/

# set environment
echo "Setup environment"
cd /afs/cern.ch/user/c/cmszcont/CMSSW_12_4_18/src/
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`
# cmsenv
cd $BASEDIR

# produce plots per Fill
echo "Produce plots per Fill"
command="python3 scripts/plot_fill.py -a $ATLAS/*.csv -c $CMS/*.csv -o $WEBDIR/Fills_overlap/ --noColorLogger --overlap --fmts png pdf"
echo $command
eval $command
command="python3 scripts/plot_fill.py -a $ATLAS/*.csv -c $CMS/*.csv -o $WEBDIR/Fills/ --noColorLogger --fmts png pdf"
echo $command
eval $command
command="python3 scripts/plot_fill.py -a $ATLAS/*.csv -c $CMS/*.csv -o $WEBDIR/Fills_refLumi/ --noColorLogger --ref-lumi --fmts png pdf"
echo $command
eval $command

# produce plots per Year
echo "Produce plots per Year"
command="python3 scripts/compute_zrate_ratio.py -a $ATLAS/*.csv -c $CMS/*.csv -o $WEBDIR --label 'Work in progress' --noColorLogger --overlap --fmts png pdf"
echo $command
eval $command

# produce plots for full Run 3
echo "Produce plots for full Run 3"
command="python3 scripts/compute_zrate_ratio.py -a $ATLASBASE/202*/*.csv -c $CMSBASE/202*/*.csv -o $WEBBASE --label 'Work in progress' --noColorLogger --overlap --fmts png pdf"
echo $command
eval $command

