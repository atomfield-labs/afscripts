"""
xTB Geometry Optimization and Frequency Driver.
Inherits from XTBCoreEngine following the Hilbert driver architecture.
"""

import os
import shutil
import sys
from pathlib import Path

from afscripts.utilities import fileutils, logger, xtb_core


log = logger.TextLogger("xtb_opt")

PROGRAM_NAME = "xTB Geometry Optimization"
JOBNAME = "xtb_opt"
FLAG_JOBNAME = "-JOBNAME"
FLAG_INPUT = "-i"
DEFAULT_JOBNAME = "xtb_opt"


class XtbOptDriver(xtb_core.XTBCoreEngine):
    """
    Driver managing standalone xTB geometry optimization calculations.
    """

    def __init__(self, options, jobname):
        """
        Initialize XtbOptDriver instance.

        :param options: Parsed command-line argument namespace.
        :type options: argparse.Namespace
        :param jobname: Job name identifier.
        :type jobname: str
        """
        super().__init__(options, jobname)
        self.input_file = None
        self.opt_output = None

    def run(self):
        """
        Standardized lifecycle execution sequence.
        """
        self.initVariables()
        self.doCalculation()
        self.exportData()

    def initVariables(self):
        """
        Initialize runtime variables and verify input structure.
        Supports explicit -i argument or automated discovery from upstream/ workspaces.
        """
        if self.options.input:
            candidate = Path(self.options.input).resolve()
            if candidate.exists():
                self.input_file = str(candidate)

        if not self.input_file:
            candidates = fileutils.find_upstream_files("*.xyz")
            if candidates:
                self.input_file = str(candidates[0].resolve())
                log(f"Auto-resolved input from upstream workspace: {self.input_file}")

        if not self.input_file:
            local = list(Path(".").glob("*.xyz"))
            if local:
                self.input_file = str(local[0].resolve())

        if not self.input_file or not os.path.exists(self.input_file):
            raise FileNotFoundError(
                "No valid input structure file found. Specify via -i <file.xyz> or place in workspace."
            )

        log(f"Starting {PROGRAM_NAME} for: {self.input_file}")

    def doCalculation(self):
        """
        Run xTB geometry optimization using XTBCoreEngine.runOpt.
        """
        self.opt_output = self.runOpt(self.input_file)

    def exportData(self):
        """
        Log final energetic summary and export standardized output files.
        """
        if not self.opt_output:
            log("Optimization did not return valid output container.")
            return

        energy_ev = self.opt_output.energy_ev
        if energy_ev is not None:
            log(f"Optimized Ground State Energy: {energy_ev:.6f} eV")

        # Standardized output geometries
        opt_fpath = self.opt_output.opt_fpath or "xtbopt.xyz"
        if os.path.exists(opt_fpath):
            output_fpath = f"{self.jobname}_optimized.xyz"
            shutil.copy2(opt_fpath, output_fpath)
            shutil.copy2(opt_fpath, "xtb_opt_optimized.xyz")
            log(f"Saved optimized geometry to: {output_fpath} & xtb_opt_optimized.xyz")

        log(f"Optimization completed successfully for {self.jobname}.")


def get_parser():
    """
    Build CLI argument parser for standalone xTB optimization.

    :return: Configured argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = xtb_core.get_parser()
    parser.add_argument(
        FLAG_INPUT,
        dest="input",
        type=str,
        default=None,
        help="Path to input structure file (.xyz, .mol, .sdf, .pdb)",
    )
    parser.add_argument(
        FLAG_JOBNAME,
        dest="jobname",
        type=str,
        default=DEFAULT_JOBNAME,
        help="Job name identifier (default: xtb_opt)",
    )
    return parser


def validate_options(options, parser):
    """
    Validate argument values.
    """
    if options.input and not os.path.exists(options.input):
        parser.error(f"Input file '{options.input}' does not exist.")


def main(args=None):
    """
    CLI entrypoint for xTB optimization driver.

    :param args: Optional argument list.
    :type args: list[str]
    """
    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    options = parser.parse_args(args)
    jobname = options.jobname if options.jobname else JOBNAME

    global log
    log = logger.TextLogger(jobname)
    validate_options(options, parser)

    driver = XtbOptDriver(options, jobname)
    driver.run()


if __name__ == "__main__":
    main()
