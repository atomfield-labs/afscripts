"""
Reaction Network Mapping Driver.
Scans MD trajectories, perceives chemical state transitions using SMILES filters,
and builds a Master Directed Reaction Graph following Hilbert driver architecture.
"""

import argparse
import glob
import io
import json
import os
import sys
from pathlib import Path

import ase.io
import networkx as nx
from networkx.readwrite import json_graph

from afscripts.analysis.rxn_event_filters import get_reaction_filter
from afscripts.scripts.xtb_nanoreactor import (
    ReactionEdge,
    ReactionNetworkBuilder,
    ReactionNode,
)
from afscripts.utilities import fileutils, logger

log = logger.TextLogger("rxn_network")

PROGRAM_NAME = "Reaction Network Mapping"
DEFAULT_JOBNAME = "rxn_network"


class RxnNetworkDriver:
    """
    Driver managing reaction network discovery from MD trajectories.
    Can analyze a single trajectory or merge across parallel MD replicates.
    """

    def __init__(self, options, jobname: str = DEFAULT_JOBNAME):
        """
        Initialize the Reaction Network Driver.

        :param options: Parsed command-line argument namespace.
        :type options: argparse.Namespace
        :param jobname: Unique job name identifier.
        :type jobname: str
        """
        self.options = options
        self.jobname = jobname
        self.trajectory_files: list[Path] = []
        self.reference_structure: ase.Atoms | None = None
        self.master_graph: nx.DiGraph | None = None
        self.discovered_species: dict[str, list[ase.Atoms]] = {}

    def run(self):
        """
        Standardized lifecycle execution sequence.
        """
        self.initVariables()
        self.doCalculation()
        self.exportData()

    def initVariables(self):
        """
        Discover and validate input trajectory files and reference reactant structure.
        Supports explicit CLI arguments or automated discovery from upstream/ workspaces.
        """
        log(f"Starting {PROGRAM_NAME} ({self.jobname})...")

        # 1. Resolve trajectory files
        raw_inputs = self.options.input or []
        if isinstance(raw_inputs, str):
            raw_inputs = [raw_inputs]

        candidate_paths: list[Path] = []
        for inp in raw_inputs:
            resolved = Path(inp).resolve()
            if resolved.exists():
                candidate_paths.append(resolved)
            else:
                # Support glob patterns like "upstream/*/*.xyz"
                expanded = [Path(p).resolve() for p in glob.glob(inp)]
                candidate_paths.extend([p for p in expanded if p.exists()])

        # Automated discovery from upstream directory if no valid inputs provided
        if not candidate_paths:
            log("No explicit input files provided. Scanning upstream/ workspaces...")
            for ext in ["*.trj", "*_traj.xyz", "*.xyz"]:
                found = fileutils.find_upstream_files(ext)
                candidate_paths.extend(found)

        # Fallback to current working directory
        if not candidate_paths:
            for ext in ["*_traj.xyz", "*.trj"]:
                candidate_paths.extend(list(Path(".").glob(ext)))

        # Deduplicate while preserving order
        seen = set()
        for p in candidate_paths:
            if p not in seen and p.is_file():
                seen.add(p)
                self.trajectory_files.append(p)

        if not self.trajectory_files:
            raise FileNotFoundError(
                "No valid trajectory files (.xyz, .trj) could be located. "
                "Specify files via -i <traj.xyz> or ensure upstream workspaces exist."
            )

        log(f"Located {len(self.trajectory_files)} trajectory file(s):")
        for idx, traj in enumerate(self.trajectory_files, 1):
            log(f"  [{idx}] {traj.name} ({traj.parent.name})")

        # 2. Resolve reference reactant structure
        if self.options.reference:
            ref_path = Path(self.options.reference).resolve()
            if ref_path.exists():
                self.reference_structure = ase.io.read(ref_path)
                log(f"Loaded reference reactant from: {ref_path.name}")

        if not self.reference_structure:
            # Check upstream for optimized reactant
            candidates = fileutils.find_upstream_files("*_optimized.xyz") or fileutils.find_upstream_files("xtbopt.xyz")
            if candidates:
                self.reference_structure = ase.io.read(candidates[0])
                log(f"Auto-resolved reference reactant from upstream: {candidates[0].name}")

        # Fallback: extract frame 0 of the first trajectory
        if not self.reference_structure:
            log("Extracting reference reactant from Frame 0 of first trajectory...")
            first_frames = ase.io.read(str(self.trajectory_files[0]), index=":")
            if not isinstance(first_frames, list):
                first_frames = [first_frames]
            self.reference_structure = first_frames[0].copy()

    def doCalculation(self):
        """
        Parse trajectory frames, perceive chemical state changes, and build master graph.
        """
        builder = ReactionNetworkBuilder(self.options)
        local_graphs = []

        for idx, traj_path in enumerate(self.trajectory_files, 1):
            log(f"Scanning trajectory [{idx}/{len(self.trajectory_files)}]: {traj_path.name}...")
            try:
                frames = ase.io.read(str(traj_path), index=":")
                if not isinstance(frames, list):
                    frames = [frames]
            except Exception as read_err:
                log(f"Warning: Failed to parse trajectory {traj_path}: {read_err}. Skipping.")
                continue

            if not frames:
                log(f"Trajectory {traj_path.name} contains 0 frames. Skipping.")
                continue

            log(f"  Loaded {len(frames)} frames. Running SMILES event detection...")
            sub_graph = builder.buildFromTrajectories(
                self.reference_structure, [frames]
            )
            local_graphs.append(sub_graph)

        self.master_graph = builder.mergeGraphs(local_graphs)
        log(
            f"Reaction Network constructed successfully: "
            f"{self.master_graph.number_of_nodes()} unique species (nodes), "
            f"{self.master_graph.number_of_edges()} reaction transitions (edges)."
        )

    def exportData(self):
        """
        Export master graph JSON, discovered product species XYZ, and summary reports.
        """
        if not self.master_graph:
            log("No reaction graph generated.")
            return

        json_out = f"{self.jobname}_network.json"
        standard_json = "raw_network.json"
        xyz_out = f"{self.jobname}_discovered_species.xyz"
        standard_xyz = "discovered_species.xyz"
        summary_out = f"{self.jobname}_summary.txt"

        export_graph = nx.DiGraph()
        all_species_frames: list[ase.Atoms] = []

        # Export nodes
        for node_id, node_attrs in self.master_graph.nodes(data=True):
            node_obj: ReactionNode = node_attrs["data"]
            num_conformers = len(node_obj.conformers)

            # Representative geometry (first conformer or geometry)
            rep_geom = node_obj.geometry or (node_obj.conformers[0] if node_obj.conformers else None)
            xyz_block = ""
            if rep_geom:
                with io.StringIO() as buf:
                    ase.io.write(buf, rep_geom, format="xyz")
                    xyz_block = buf.getvalue()

                # Annotate conformer for multi-structure export
                tagged_atoms = rep_geom.copy()
                tagged_atoms.info["smiles"] = node_id
                tagged_atoms.info["node_type"] = node_obj.node_type
                tagged_atoms.info["conformer_count"] = num_conformers
                all_species_frames.append(tagged_atoms)

            export_graph.add_node(
                node_id,
                node_type=node_obj.node_type,
                conformer_count=num_conformers,
                structure=xyz_block,
            )

        # Export edges
        for source, target, edge_attrs in self.master_graph.edges(data=True):
            edge_obj: ReactionEdge = edge_attrs.get("data", ReactionEdge())
            export_graph.add_edge(
                source,
                target,
                Ea=edge_obj.ea,
                ts_structure=edge_obj.ts_structure,
            )

        # 1. Write Network JSON
        graph_data = json_graph.node_link_data(export_graph)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
        with open(standard_json, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)

        # 2. Write Discovered Species XYZ
        if all_species_frames:
            with open(xyz_out, "w", encoding="utf-8") as f:
                for atoms in all_species_frames:
                    smi = atoms.info.get("smiles", "Unknown")
                    count = atoms.info.get("conformer_count", 1)
                    with io.StringIO() as buf:
                        ase.io.write(buf, atoms, format="xyz")
                        lines = buf.getvalue().strip().split("\n")
                    lines[1] = f"SMILES:{smi} Conformers:{count}"
                    f.write("\n".join(lines) + "\n")

            # Also create standard discovered_species.xyz alias
            with open(standard_xyz, "w", encoding="utf-8") as f:
                for atoms in all_species_frames:
                    smi = atoms.info.get("smiles", "Unknown")
                    count = atoms.info.get("conformer_count", 1)
                    with io.StringIO() as buf:
                        ase.io.write(buf, atoms, format="xyz")
                        lines = buf.getvalue().strip().split("\n")
                    lines[1] = f"SMILES:{smi} Conformers:{count}"
                    f.write("\n".join(lines) + "\n")

        # 3. Write Text Summary
        with open(summary_out, "w", encoding="utf-8") as f:
            f.write("====================================================\n")
            f.write(f"  AtomField Reaction Network Discovery Summary\n")
            f.write("====================================================\n")
            f.write(f"Total Trajectories Analyzed: {len(self.trajectory_files)}\n")
            f.write(f"Discovered Unique Species (Nodes): {self.master_graph.number_of_nodes()}\n")
            f.write(f"Discovered Physical Transitions (Edges): {self.master_graph.number_of_edges()}\n\n")
            f.write("Species List:\n")
            for idx, (node_id, node_attrs) in enumerate(self.master_graph.nodes(data=True), 1):
                node_obj = node_attrs["data"]
                f.write(f"  [{idx}] {node_id} ({node_obj.node_type}, {len(node_obj.conformers)} conformers)\n")
            f.write("\nReaction Pathways:\n")
            for idx, (src, dst) in enumerate(self.master_graph.edges(), 1):
                f.write(f"  [{idx}] {src}  --->  {dst}\n")

        log(f"Export complete:")
        log(f"  - Network Graph: {json_out} & {standard_json}")
        log(f"  - Discovered Species: {xyz_out} & {standard_xyz}")
        log(f"  - Summary Report: {summary_out}")


def get_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser for standalone reaction network mapper.

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
        nargs="*",
        default=None,
        help="Path(s) or glob pattern to input MD trajectory file(s) (.xyz, .trj)",
    )
    parser.add_argument(
        "-ref",
        dest="reference",
        type=str,
        default=None,
        help="Path to starting reference reactant structure file (.xyz)",
    )
    parser.add_argument(
        "-c",
        dest="charge",
        type=int,
        default=0,
        help="Total molecular charge for RDKit valence perception",
    )
    parser.add_argument(
        "--rxn_filter",
        dest="rxn_filter",
        type=str,
        default="smiles",
        help="Filter algorithm for state differentiation ('smiles')",
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

    driver = RxnNetworkDriver(options, options.jobname)
    driver.run()


if __name__ == "__main__":
    main(sys.argv[1:])
