"""
Transition State (TS) Search Driver.
Computes Minimum Energy Paths (MEP), saddle points, and activation barriers (Ea)
using Sella and Nudged Elastic Band (NEB) with SevenNet following Hilbert architecture.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import ase.io
import networkx as nx
from networkx.readwrite import json_graph

from afscripts.utilities import fileutils, logger
from afscripts.utilities.rdpattern import atoms_to_rdkit_mol, write_sdf
from afscripts.utilities.ts_searchers import get_ts_searcher

log = logger.TextLogger("ts_search")

PROGRAM_NAME = "Transition State Search"
DEFAULT_JOBNAME = "ts_search"


class TSSearchDriver:
    """
    Driver managing Transition State (TS) optimization and activation barrier calculations.
    Supports network-wide edge scanning or direct two-structure (reactant -> product) runs.
    """

    def __init__(self, options, jobname: str = DEFAULT_JOBNAME):
        """
        Initialize the TS Search Driver.

        :param options: Parsed command-line argument namespace.
        :type options: argparse.Namespace
        :param jobname: Unique job name identifier.
        :type jobname: str
        """
        self.options = options
        self.jobname = jobname
        self.network_file: Path | None = None
        self.structures_file: Path | None = None
        self.reactant_file: Path | None = None
        self.product_file: Path | None = None
        self.master_graph: nx.DiGraph | None = None
        self.species_geometries: dict[str, ase.Atoms] = {}
        self.ts_results: list[dict] = []

    def run(self):
        """
        Standardized lifecycle execution sequence.
        """
        self.initVariables()
        self.doCalculation()
        self.exportData()

    def initVariables(self):
        """
        Discover input reaction network JSON and ground-state geometry files,
        or direct reactant/product pair from CLI or upstream/ workspaces.
        """
        log(f"Starting {PROGRAM_NAME} ({self.jobname})...")

        # 1. Direct Reactant/Product mode
        if self.options.reactant and self.options.product:
            r_path = Path(self.options.reactant).resolve()
            p_path = Path(self.options.product).resolve()
            if r_path.exists() and p_path.exists():
                self.reactant_file = r_path
                self.product_file = p_path
                log(f"Operating in Direct Pair mode:")
                log(f"  Reactant: {r_path.name}")
                log(f"  Product:  {p_path.name}")
                return

        # 2. Network mode - Resolve network JSON
        if self.options.network:
            c = Path(self.options.network).resolve()
            if c.exists():
                self.network_file = c

        if not self.network_file:
            candidates = fileutils.find_upstream_files("raw_network.json") or fileutils.find_upstream_files("*_network.json")
            if candidates:
                self.network_file = candidates[0].resolve()

        if not self.network_file:
            local = list(Path(".").glob("raw_network.json")) or list(
                Path(".").glob("*_network.json")
            )
            if local:
                self.network_file = local[0].resolve()

        # 3. Network mode - Resolve refined structures
        if self.options.structures:
            c = Path(self.options.structures).resolve()
            if c.exists():
                self.structures_file = c

        if not self.structures_file:
            candidates = (
                fileutils.find_upstream_files("refined_species.xyz")
                or fileutils.find_upstream_files("*_refined_species.xyz")
                or fileutils.find_upstream_files("discovered_species.xyz")
            )
            if candidates:
                self.structures_file = candidates[0].resolve()

        if not self.structures_file:
            local = list(Path(".").glob("refined_species.xyz")) or list(
                Path(".").glob("discovered_species.xyz")
            )
            if local:
                self.structures_file = local[0].resolve()

        if self.network_file and self.structures_file:
            log(f"Operating in Reaction Network mode:")
            log(f"  Network JSON: {self.network_file.name}")
            log(f"  Structures:   {self.structures_file.name}")
            with open(self.network_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.master_graph = json_graph.node_link_graph(data)

            structs = ase.io.read(str(self.structures_file), index=":")
            if not isinstance(structs, list):
                structs = [structs]
            for s in structs:
                smi = s.info.get("smiles")
                if not smi:
                    # Parse from comment line
                    smi = s.get_chemical_formula()
                self.species_geometries[smi] = s
        else:
            raise FileNotFoundError(
                "Could not locate input files for TS Search. Provide either "
                "-reactant <r.xyz> -product <p.xyz> OR -network <net.json> -structures <structs.xyz>."
            )

    def doCalculation(self):
        """
        Execute Sella + NEB transition state search on reaction edges or standalone pair.
        """
        ts_method = getattr(self.options, "ts_search", "sella_neb")
        log(f"Initializing TS searcher: {ts_method}...")
        searcher = get_ts_searcher(ts_method, self.options)

        # Case A: Direct Reactant -> Product pair
        if self.reactant_file and self.product_file:
            r_atoms = ase.io.read(str(self.reactant_file))
            p_atoms = ase.io.read(str(self.product_file))
            log("Searching transition state between reactant and product...")
            try:
                ts_atoms, ea = searcher.findTransitionState(r_atoms, p_atoms)
                log(f"  TS Located! Activation Barrier Ea: {ea:.4f} eV")
                self.ts_results.append(
                    {
                        "source": self.reactant_file.stem,
                        "target": self.product_file.stem,
                        "ea": ea,
                        "ts_atoms": ts_atoms,
                    }
                )
            except Exception as ts_err:
                log(f"  TS Search failed: {ts_err}")
            return

        # Case B: Network edges
        if not self.master_graph:
            return

        edges = list(self.master_graph.edges())
        log(f"Evaluating {len(edges)} reaction edge(s) in the network...")

        for idx, (source, target) in enumerate(edges, 1):
            log(f"Edge [{idx}/{len(edges)}]: {source}  --->  {target}...")

            source_atoms = self.species_geometries.get(source)
            target_atoms = self.species_geometries.get(target)

            if source_atoms is None or target_atoms is None:
                log(f"  Skipping edge: geometry not found for {source} or {target}.")
                continue

            try:
                ts_atoms, ea = searcher.findTransitionState(source_atoms, target_atoms)
                log(f"  Saddle point converged! Ea = {ea:.4f} eV")
                with io.StringIO() as buf:
                    ase.io.write(buf, ts_atoms, format="xyz")
                    ts_xyz = buf.getvalue()

                self.master_graph.edges[source, target]["Ea"] = ea
                self.master_graph.edges[source, target]["ts_structure"] = ts_xyz

                self.ts_results.append(
                    {
                        "source": source,
                        "target": target,
                        "ea": ea,
                        "ts_atoms": ts_atoms,
                    }
                )
            except Exception as ts_err:
                log(f"  Warning: TS search failed for {source} -> {target}: {ts_err}")

    def exportData(self):
        """
        Export final annotated reaction network JSON, TS structures XYZ,
        RDKit SDF database, and summary reports.
        """
        json_out = f"{self.jobname}_final_network.json"
        standard_json = "final_network.json"
        xyz_out = f"{self.jobname}_ts_geometries.xyz"
        standard_xyz = "ts_geometries.xyz"
        sdf_out = f"{self.jobname}_reaction_summary.sdf"
        summary_out = f"{self.jobname}_summary.txt"

        # 1. Write Updated Network JSON
        if self.master_graph:
            graph_data = json_graph.node_link_data(self.master_graph)
            for path in [json_out, standard_json]:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, indent=2)

        # 2. Write Transition State Geometries XYZ
        if self.ts_results:
            for path in [xyz_out, standard_xyz]:
                with open(path, "w", encoding="utf-8") as f:
                    for item in self.ts_results:
                        ts_atoms = item["ts_atoms"]
                        ea = item["ea"]
                        src = item["source"]
                        dst = item["target"]
                        with io.StringIO() as buf:
                            ase.io.write(buf, ts_atoms, format="xyz")
                            lines = buf.getvalue().strip().split("\n")
                        lines[1] = f"TS:{src}->{dst} Ea:{ea:.4f}eV"
                        f.write("\n".join(lines) + "\n")

        # 3. Write RDKit SDF Database of species
        if self.species_geometries:
            mols = []
            props = []
            charge = getattr(self.options, "charge", 0)
            for smi, atoms in self.species_geometries.items():
                try:
                    rdmol = atoms_to_rdkit_mol(atoms, charge=charge)
                    if rdmol:
                        mols.append(rdmol)
                        props.append({"SMILES": smi, "Formula": atoms.get_chemical_formula()})
                except Exception:
                    pass
            if mols:
                write_sdf(mols, sdf_out, properties_list=props)
                write_sdf(mols, "reaction_summary.sdf", properties_list=props)

        # 4. Write Summary Report
        with open(summary_out, "w", encoding="utf-8") as f:
            f.write("====================================================\n")
            f.write(f"  AtomField Transition State Search Summary\n")
            f.write("====================================================\n")
            f.write(f"TS Engine: {getattr(self.options, 'ts_engine', 'sevennet')}\n")
            f.write(f"NEB Engine: {getattr(self.options, 'neb_engine', 'sevennet')}\n")
            f.write(f"Total TS Searches Evaluated: {len(self.ts_results)}\n\n")
            f.write("Calculated Activation Barriers (Ea):\n")
            for idx, item in enumerate(self.ts_results, 1):
                f.write(
                    f"  [{idx}] {item['source']}  --->  {item['target']} | Ea = {item['ea']:.4f} eV\n"
                )

        log("Export complete:")
        if self.master_graph:
            log(f"  - Final Network: {json_out} & {standard_json}")
        if self.ts_results:
            log(f"  - TS Geometries: {xyz_out} & {standard_xyz}")
        log(f"  - Chemical SDF:  {sdf_out}")
        log(f"  - Summary Report: {summary_out}")


def get_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser for Transition State Search driver.

    :return: Configured argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        prefix_chars="-",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-network",
        dest="network",
        type=str,
        default=None,
        help="Path to reaction network JSON file (raw_network.json)",
    )
    parser.add_argument(
        "-structures",
        dest="structures",
        type=str,
        default=None,
        help="Path to refined ground-state structures file (refined_species.xyz)",
    )
    parser.add_argument(
        "-reactant",
        dest="reactant",
        type=str,
        default=None,
        help="Path to single reactant structure (.xyz) for direct pairwise search",
    )
    parser.add_argument(
        "-product",
        dest="product",
        type=str,
        default=None,
        help="Path to single product structure (.xyz) for direct pairwise search",
    )
    parser.add_argument(
        "--ts_search",
        dest="ts_search",
        type=str,
        default="sella_neb",
        help="TS search algorithm ('sella_neb')",
    )
    parser.add_argument(
        "--neb_engine",
        dest="neb_engine",
        type=str,
        default="sevennet",
        help="Energy & gradient engine for NEB path exploration",
    )
    parser.add_argument(
        "--ts_engine",
        dest="ts_engine",
        type=str,
        default="sevennet",
        help="Energy & gradient engine for Sella saddle point refinement",
    )
    parser.add_argument(
        "--nimages",
        dest="nimages",
        type=int,
        default=10,
        help="Number of intermediate images in NEB band",
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

    driver = TSSearchDriver(options, options.jobname)
    driver.run()


if __name__ == "__main__":
    main(sys.argv[1:])
