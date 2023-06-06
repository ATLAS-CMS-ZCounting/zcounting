import argparse
from common.logging import child_logger
log = child_logger(__name__)


def parser(pars=None):
    if pars is None:
        pars = argparse.ArgumentParser()

    pars.add_argument("-v", "--verbose", type=int, default=3, choices=[0,1,2,3,4],
                        help="Set verbosity level with logging, the larger the more verbose")
    pars.add_argument("-c", "--cms-csv", type=str, nargs="+", default=["resources/zrate_cms_2p4.csv"], 
                        help="Input csv file(s) with z boson rates from CMS")
    pars.add_argument("-a", "--atlas-csv", type=str, nargs="+", default=["resources/zrate_cms_2p4.csv"], 
                        help="Input csv file(s) with z boson rates from CMS")
    pars.add_argument("-o", "--outputDir", default="./Test", 
                        help="Output directory")
    pars.add_argument("-l", "--label", default="Internal", choices=["Internal", "Work in progress", "Preliminary", ""],
                        help="Label attached to the ATLAS+CMS label on the plots")
    pars.add_argument("--overlap", default=False, action="store_true",
                        help="Remove measurements that do not overlap between ATLAS and CMS")
    pars.add_argument("--threshold-outlier", default=0.3, type=float,
                        help="Remove measurements where the Z boson rate luminosity and reference luminosity disagree by the specified number.")
    return pars
