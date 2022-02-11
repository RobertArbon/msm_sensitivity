"""
The CLI. This adaptes the Osprey CLI:
https://github.com/msmbuilder/osprey/blob/master/osprey/cli/main.py
"""
from __future__ import print_function, absolute_import, division

import sys
import argparse

from msmsense import sample_hps
from msmsense import bootstrap
# from . import parser_dump
# from . import parser_skeleton
# from . import parser_worker
# from . import parser_plot
# from . import parser_currentbest


def main():
    help = 'MSM-sense is a tool for testing the sensitivity of MSM hyperparameters'
    p = argparse.ArgumentParser(description=help)
    # p.add_argument(
    #     '-V', '--version',
    #     action='version',
    #     version='msmsense %s' % __version__,
    # )
    sub_parsers = p.add_subparsers(
        metavar='command',
        dest='cmd',
    )

    sample_hps.configure_parser(sub_parsers)  # Docs say this is correct type.
    bootstrap_cmatrices.configure_parser(sub_parsers)  # Docs say this is correct type.

    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = p.parse_args()
    args_func(args, p)


def args_func(args, p):
    try:
        args.func(args, p)
    except RuntimeError as e:
        sys.exit("Error: %s" % e)
    except Exception as e:
        if e.__class__.__name__ not in ('ScannerError', 'ParserError'):
            message = """\
An unexpected error has occurred with MSM Sense please
consider sending the following traceback to the MSM Sense GitHub issue tracker at:
        https://github.com/RobertArbon/msm_sensitivity/issues
The error that cause this message was: 
"""
            print(message, e, file=sys.stderr)
            # print(message % __version__, file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())