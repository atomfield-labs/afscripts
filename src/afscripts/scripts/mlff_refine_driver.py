"""
Product Conformer Refinement Driver using SevenNet Neural Network Potential.
Optimizes raw sampled conformers from MD trajectories to find true local ground states
and accurate potential energies following the Hilbert driver architecture.
"""

import argparse
import csv
import io
import os
import sys
from pathlib import Path

import ase.io

from afscripts.utilities import fileutils, logger
from afscripts.utilities.product_refiners import get_refiner
from afscripts.utilities.rdpattern import get_canonical_smiles

log = logger.TextLogger("mlff_refine")

PROGRAM_NAME = "SevenNet Conformer Refinement"
DEFAULT_JOBNAME = "mlff_refine"


class MLFFRefineDriver:
    """
    Driver managing conformer geometry optimization and ground-state selection
    using SevenNet Universal Machine-Learning Interatomic Potential (MLIP).
    """

    def __init__(self, options, jobname: str = DEFAULT_JOBNAME):
        """
        Initialize the MLFF Refine Driver.

        :param options: Parsed command-line argument namespace.
        :type options: argparse.Namespace
        :param jobname: Unique job name identifier.
        :type jobname: str
        """
        self.options = options
        self.jobname = jobname
        self.input_file: Path | None = None
        self.input_structures: list[ase.Atoms] = []
        self.refiner = None
        self.refined_results: list[dict] = []

    def run(self):
        """
        Standardized lifecycle execution sequence.
        """
        self.initVariables()
        self.doCalculation()
        self.exportData()

    def initVariables(self):
        """
        Resolve input coordinate file and initialize SevenNet refiner engine.
        Supports explicit -i argument or automated discovery from upstream/ workspaces.
        """
        log(f"Starting {PROGRAM_NAME} ({self.jobname})...")

        # 1. Resolve input file
        if self.options.input:
            candidate = Path(self.options.input).resolve()
            if candidate.exists():
                self.input_file = candidate

        if not self.input_file:
            # Auto-discover from upstream workspaces
            candidates = (
                fileutils.find_upstream_files("discovered_species.xyz")
                or fileutils.find_upstream_files("*_discovered_species.xyz")
                or fileutils.find_upstream_files("*_optimized.xyz")
                or fileutils.find_upstream_files("*.xyz")
            )
            if candidates:
                self.input_file = candidates[0].resolve()
                log(f"Auto-resolved input from upstream workspace: {self.input_file.name}")

        if not self.input_file:
            # Fallback to current directory
            local_candidates = list(Path(".").glob("discovered_species.xyz")) or list(
                Path(".").glob("*.xyz")
            )
            if local_candidates:
                self.input_file = local_candidates[0].resolve()

        if not self.input_file or not self.input_file.exists():
            raise FileNotFoundError(
                "No valid input structure file (.xyz, .mol, .sdf) could be found. "
                "Specify via -i <file.xyz> or ensure upstream workspaces exist."
            )

        log(f"Loading structures from: {self.input_file}")
        raw_atoms = ase.io.read(str(self.input_file), index=":")
        if not isinstance(raw_atoms, list):
            raw_atoms = [raw_atoms]
        self.input_structures = raw_atoms
        log(f"Loaded {len(self.input_structures)} candidate structure(s) for refinement.")

        # 2. Instantiate refiner engine
        engine_name = getattr(self.options, "engine", "sevennet")
        model_name = getattr(self.options, "model", "7net-0")
        log(f"Initializing refinement engine: {engine_name} (model: {model_name})...")
        self.refiner = get_refiner(engine_name, model=model_name)

    def doCalculation(self):
        """
        Execute geometry relaxation on all input structures using SevenNet MLIP.
        """
        fmax = getattr(self.options, "fmax", 0.01)
        steps = getattr(self.options, "steps", 250)
        charge = getattr(self.options, "charge", 0)

        species_groups: dict[str, list[tuple[ase.Atoms, float]]] = {}

        for idx, atoms in enumerate(self.input_structures, 1):
            formula = atoms.get_chemical_formula()
            log(f"Refining structure [{idx}/{len(self.input_structures)}] ({formula})...")

            try:
                relaxed = self.refiner.runOptimization(atoms, fmax=fmax, steps=steps)
                energy = float(relaxed.get_potential_energy())
                log(f"  Converged! Energy: {energy:.6f} eV")
            except Exception as opt_err:
                log(f"  Warning: Optimization failed: {opt_err}. Retaining unrelaxed energy.")
                relaxed = atoms.copy()
                try:
                    energy = float(self.refiner.runSCF(relaxed).get_potential_energy())
                except Exception:
                    energy = 0.0

            # Perceive SMILES identifier
            try:
                smiles = get_canonical_smiles(relaxed, charge=charge)
            except Exception:
                smiles = atoms.info.get("smiles", formula)

            if smiles not in species_groups:
                species_groups[smiles] = []
            species_groups[smiles].append((relaxed, energy))

        # Select lowest-energy conformer for each unique species
        for smiles, conformers in species_groups.items():
            conformers.sort(key=lambda item: item[1])
            best_geom, best_energy = conformers[0]
            self.refined_results.append(
                {
                    "smiles": smiles,
                    "formula": best_geom.get_chemical_formula(),
                    "energy_ev": best_energy,
                    "geometry": best_geom,
                    "conformer_count": len(conformers),
                }
            )

        # Sort all discovered species by ground-state potential energy
        self.refined_results.sort(key=lambda r: r["energy_ev"])
        log(
            f"Refinement complete for {len(self.refined_results)} unique chemical species."
        )

    def exportData(self):
        """
        Export refined 0K ground state geometries (.xyz), conformer energy table (.csv),
        and summary text reports.
        """
        if not self.refined_results:
            log("No refined results to export.")
            return

        xyz_out = f"{self.jobname}_refined_species.xyz"
        standard_xyz = "refined_species.xyz"
        csv_out = f"{self.jobname}_conformer_energies.csv"
        standard_csv = "conformer_energies.csv"
        summary_out = f"{self.jobname}_summary.txt"

        # 1. Write Multi-Structure XYZ with energy comments
        with open(xyz_out, "w", encoding="utf-8") as f:
            for item in self.refined_results:
                atoms = item["geometry"]
                energy = item["energy_ev"]
                smi = item["smiles"]
                with io.StringIO() as buf:
                    ase.io.write(buf, atoms, format="xyz")
                    lines = buf.getvalue().strip().split("\n")
                lines[1] = f"SMILES:{smi} Energy:{energy:.6f}eV Formula:{item['formula']}"
                f.write("\n".join(lines) + "\n")

        with open(standard_xyz, "w", encoding="utf-8") as f:
            for item in self.refined_results:
                atoms = item["geometry"]
                energy = item["energy_ev"]
                smi = item["smiles"]
                with io.StringIO() as buf:
                    ase.io.write(buf, atoms, format="xyz")
                    lines = buf.getvalue().strip().split("\n")
                lines[1] = f"SMILES:{smi} Energy:{energy:.6f}eV Formula:{item['formula']}"
                f.write("\n".join(lines) + "\n")

        # 2. Write CSV Energy Table
        headers = ["Rank", "SMILES", "Formula", "Energy_eV", "Conformers_Sampled"]
        rows = [
            [
                idx,
                item["smiles"],
                item["formula"],
                f"{item['energy_ev']:.6f}",
                item["conformer_count"],
            ]
            for idx, item in enumerate(self.refined_results, 1)
        ]

        for filepath in [csv_out, standard_csv]:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

        # 3. Write Summary Report
        with open(summary_out, "w", encoding="utf-8") as f:
            f.write("====================================================\n")
            f.write(f"  AtomField SevenNet Conformer Refinement Summary\n")
            f.write("====================================================\n")
            f.write(f"Refinement Engine: {getattr(self.options, 'engine', 'sevennet')}\n")
            f.write(f"Model: {getattr(self.options, 'model', '7net-0')}\n")
            f.write(f"Unique Species Refined: {len(self.refined_results)}\n\n")
            f.write("Refined Species Energy Hierarchy (0K Ground State):\n")
            for idx, item in enumerate(self.refined_results, 1):
                f.write(
                    f"  [{idx}] {item['smiles']} ({item['formula']}): "
                    f"{item['energy_ev']:.6f} eV (sampled {item['conformer_count']} conformers)\n"
                )

        log(f"Export complete:")
        log(f"  - Refined Geometries: {xyz_out} & {standard_xyz}")
        log(f"  - Conformer Energy Table: {csv_out} & {standard_csv}")
        log(f"  - Summary Report: {summary_out}")


def get_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser for SevenNet conformer refinement driver.

    :return: Configured argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        prefix_chars="-",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        dest="input",
        type=str,
        default=None,
        help="Path to input structure or multi-structure file (.xyz, .mol, .sdf)",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        type=str,
        default="sevennet",
        help="Refinement engine backend ('sevennet', 'xtb')",
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default="7net-0",
        help="SevenNet model checkpoint name",
    )
    parser.add_argument(
        "--fmax",
        dest="fmax",
        type=float,
        default=0.01,
        help="Maximum force convergence threshold in eV/Å",
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        type=int,
        default=250,
        help="Maximum number of BFGS optimization steps",
    )
    parser.add_argument(
        "-c",
        dest="charge",
        type=int,
        default=0,
        help="Total molecular charge",
    )
    parser.add_argument(
        "-JOBNAME",
        dest="jobname",
        type=str,
        default=DEFAULT_JOBNAME,
        help="Job name identifier for output artifact naming",
    )
    return parser


def validate_options(options, parser):
    """
    Validate runtime options.
    """
    pass


def main(args=None):
    """
    Standalone CLI entrypoint.
    """
    if args is None:
        args = sys.argv[1:]
    parser = get_parser()
    options = parser.parse_args(args)
    validate_options(options, parser)

    driver = MLFFRefineDriver(options, options.jobname)
    driver.run()


if __name__ == "__main__":
    main(sys.argv[1:])
