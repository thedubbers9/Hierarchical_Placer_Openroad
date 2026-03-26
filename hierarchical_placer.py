"""
Hierarchical Macro Placer

Implements a recursive placement algorithm for macro blocks:
  - If the number of macros exceeds threshold N, use greedy connectivity-based
    clustering to partition into D groups, then recurse on each group.
  - If the number of macros is <= N, enumerate all permutations of macros into
    D grid slots and pick the arrangement with minimum HPWL.

The placer operates on the functional-unit-level graph (one node per Add/Mult/Mux/etc.)
and produces (x, y) coordinates for each macro, which are written to a TCL script
that can be sourced by OpenROAD in place of rtl_macro_placer.
"""

import logging
import math
import itertools
import heapq
import copy
from typing import Dict, List, Tuple, Set, Optional

import networkx as nx

logger = logging.getLogger(__name__)

DEBUG = True

def debug_print(msg):
    if DEBUG:
        logger.info(msg)


# ──────────────────────────────────────────────────────────────
#  Default parameters
# ──────────────────────────────────────────────────────────────

## Recursion threshold: if #macros <= N, enumerate permutations
DEFAULT_N = 4

## Number of groups / slots for clustering and enumeration
DEFAULT_D = 4

## Aspect ratio bounds for sub-regions (height / width)
DEFAULT_MIN_ASPECT_RATIO = 0.33
DEFAULT_MAX_ASPECT_RATIO = 3.0

## Gap between macros (microns) – used as minimum spacing
DEFAULT_MACRO_GAP = 10.0
DEFAULT_ENABLE_SMALL_MACRO_HALO = True
DEFAULT_MAX_SMALL_MACRO_HALO = 20.0
DEFAULT_SMALL_MACRO_HALO_MEDIAN_FACTOR = 1.0


# ──────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────

class Region:
    """Axis-aligned rectangular region in microns."""

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    def __repr__(self):
        return f"Region({self.x_min:.1f}, {self.y_min:.1f}, {self.x_max:.1f}, {self.y_max:.1f})"


# ──────────────────────────────────────────────────────────────
#  HierarchicalPlacer
# ──────────────────────────────────────────────────────────────

class HierarchicalPlacer:
    """
    Hierarchical macro placer.

    Parameters
    ----------
    graph : nx.DiGraph
        Functional-unit-level graph.  Each node represents one macro
        (Add, Mult, Mux, Call, etc.).  Edges represent data-flow
        connectivity between functional units.
    macro_size_dict : dict[str, (float, float)]
        Mapping from macro LEF name -> (width_um, height_um).
    node_to_macro_name : dict[str, str]
        Mapping from graph node name -> macro LEF name (used to look up sizes).
    core_area : tuple (x_min, y_min, x_max, y_max)
        The placement region in microns.
    N : int
        Recursion threshold.  Groups with <= N macros are solved by
        exhaustive permutation enumeration.
    D : int
        Number of groups for clustering and number of slots for
        enumeration.
    min_aspect_ratio : float
        Minimum allowed aspect ratio (height / width) for sub-regions.
    max_aspect_ratio : float
        Maximum allowed aspect ratio (height / width) for sub-regions.
    macro_gap : float
        Minimum gap between placed macros (microns).
    manufacturing_grid : float
        Manufacturing grid for snapping coordinates (microns).
        Defaults to 0.005 (FreePDK45).
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        macro_size_dict: Dict[str, Tuple[float, float]],
        node_to_macro_name: Dict[str, str],
        core_area: Tuple[float, float, float, float],
        N: int = DEFAULT_N,
        D: int = DEFAULT_D,
        min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO,
        max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
        macro_gap: float = DEFAULT_MACRO_GAP,
        enable_small_macro_halo: bool = DEFAULT_ENABLE_SMALL_MACRO_HALO,
        max_small_macro_halo: float = DEFAULT_MAX_SMALL_MACRO_HALO,
        small_macro_halo_median_factor: float = DEFAULT_SMALL_MACRO_HALO_MEDIAN_FACTOR,
        manufacturing_grid: float = 0.005,
    ):
        self.graph = graph
        self.macro_size_dict = macro_size_dict
        self.node_to_macro_name = node_to_macro_name
        self.core_region = Region(*core_area)
        self.N = N
        self.D = D
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.macro_gap = macro_gap
        self.enable_small_macro_halo = enable_small_macro_halo
        self.max_small_macro_halo = max_small_macro_halo
        self.small_macro_halo_median_factor = small_macro_halo_median_factor
        self.manufacturing_grid = manufacturing_grid

        # Pre-compute node areas for fast lookup
        self._node_area: Dict[str, float] = {}
        for node in self.graph.nodes():
            macro_name = self.node_to_macro_name.get(node)
            if macro_name and macro_name in self.macro_size_dict:
                w, h = self.macro_size_dict[macro_name]
                self._node_area[node] = w * h
            else:
                self._node_area[node] = 0.0

        # Pre-compute node sizes
        self._node_size: Dict[str, Tuple[float, float]] = {}
        for node in self.graph.nodes():
            macro_name = self.node_to_macro_name.get(node)
            if macro_name and macro_name in self.macro_size_dict:
                self._node_size[node] = self.macro_size_dict[macro_name]
            else:
                self._node_size[node] = (0.0, 0.0)

        nonzero_areas = sorted(a for a in self._node_area.values() if a > 0.0)
        if nonzero_areas:
            self._median_macro_area = nonzero_areas[len(nonzero_areas) // 2]
        else:
            self._median_macro_area = 0.0

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def place(self) -> Dict[str, Tuple[float, float]]:
        """
        Run the hierarchical placement algorithm.

        Returns
        -------
        positions : dict[str, (float, float)]
            Mapping from graph node name -> (x, y) lower-left corner in microns.
        """
        nodes = [n for n in self.graph.nodes() if self._node_area.get(n, 0) > 0]

        if len(nodes) == 0:
            debug_print("No macros to place.")
            return {}

        debug_print(f"Placing {len(nodes)} macros in region {self.core_region}")
        positions = self._place_recursive(nodes, self.core_region, depth=0)

        # Snap to manufacturing grid
        snapped = {}
        for node, (x, y) in positions.items():
            snapped[node] = (self._snap(x), self._snap(y))

        debug_print(f"Placement complete. {len(snapped)} macros placed.")
        return snapped

    def write_placement_tcl(
        self,
        positions: Dict[str, Tuple[float, float]],
        node_to_component_num: Dict[str, str],
        output_path: str,
    ):
        """
        Write a TCL script with fixed macro placement commands.

        Parameters
        ----------
        positions : dict[str, (float, float)]
            Node name -> (x, y) coordinates in microns.
        node_to_component_num : dict[str, str]
            Node name -> DEF component ID (e.g. "_001_").
        output_path : str
            File path for the output TCL script.
        """
        with open(output_path, "w") as f:
            f.write("# Hierarchical macro placement generated by hierarchical_placer.py\n")
            f.write("# This file is sourced by codesign_flow.tcl\n\n")

            # Set up DB access (these variables may not be available yet in the flow)
            f.write("set _hp_db [ord::get_db]\n")
            f.write("set _hp_block [[$_hp_db getChip] getBlock]\n")
            f.write("set _hp_tech [$_hp_db getTech]\n")
            f.write("set _hp_dbu [$_hp_tech getDbUnitsPerMicron]\n\n")

            for node, (x, y) in positions.items():
                component_id = node_to_component_num.get(node)
                if component_id is None:
                    debug_print(f"Warning: no component ID for node {node}, skipping TCL placement.")
                    continue

                f.write(f'# Node: {node}\n')
                f.write(f'set _hp_inst [$_hp_block findInst "{component_id}"]\n')
                f.write(f'if {{$_hp_inst != "NULL"}} {{\n')
                f.write(f'  set _hp_x [expr {{round({x} * $_hp_dbu)}}]\n')
                f.write(f'  set _hp_y [expr {{round({y} * $_hp_dbu)}}]\n')
                f.write(f'  $_hp_inst setLocation $_hp_x $_hp_y\n')
                f.write(f'  $_hp_inst setPlacementStatus "FIRM"\n')
                f.write(f'}}\n\n')

            f.write('puts "Hierarchical placement applied."\n')

        debug_print(f"Wrote placement TCL to {output_path}")

    # ──────────────────────────────────────────────────────────
    #  Core recursive algorithm
    # ──────────────────────────────────────────────────────────

    def _place_recursive(
        self,
        nodes: List[str],
        region: Region,
        depth: int,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Recursively place macros within a region.

        - Base case (len(nodes) <= N): exhaustive permutation of slot assignments.
        - Recursive case: cluster into D groups, allocate sub-regions, recurse.
        """
        indent = "  " * depth
        debug_print(f"{indent}place_recursive: {len(nodes)} nodes in {region}")

        if len(nodes) == 0:
            return {}

        if len(nodes) == 1:
            # Trivial: center the single macro in the region
            node = nodes[0]
            w, h = self._node_size[node]
            halo = self._small_macro_halo(node)
            ew = w + 2.0 * halo
            eh = h + 2.0 * halo
            x = region.cx - ew / 2.0 + halo
            y = region.cy - eh / 2.0 + halo
            # Clamp to region
            x = max(region.x_min + halo, min(x, region.x_max - w - halo))
            y = max(region.y_min + halo, min(y, region.y_max - h - halo))
            return {node: (x, y)}

        if len(nodes) <= self.N:
            return self._enumerate_placements(nodes, region, depth)
        else:
            return self._cluster_and_recurse(nodes, region, depth)

    # ──────────────────────────────────────────────────────────
    #  Base case: exhaustive enumeration
    # ──────────────────────────────────────────────────────────

    def _enumerate_placements(
        self,
        nodes: List[str],
        region: Region,
        depth: int,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Enumerate all permutations of nodes into grid slots.
        Pick the permutation with minimum HPWL.
        """
        indent = "  " * depth
        n = len(nodes)

        # Create grid slots within the region
        slots = self._compute_grid_slots(nodes, region)

        if len(slots) < n:
            debug_print(f"{indent}Warning: only {len(slots)} slots for {n} nodes, falling back to row packing.")
            return self._pack_in_rows(nodes, region)

        # Enumerate all permutations of slot assignments
        best_hpwl = float("inf")
        best_assignment = None

        # For efficiency, limit permutation count.  If n > 8, this would be 8! = 40320 which
        # is already borderline. The N parameter should keep this in check.
        slot_indices = list(range(len(slots)))

        for perm in itertools.permutations(slot_indices[:n]):
            # Assign each node to a slot
            candidate = {}
            valid = True
            for i, node in enumerate(nodes):
                slot_idx = perm[i]
                sx, sy, sw, sh = slots[slot_idx]
                nw, nh = self._node_size[node]
                halo = self._small_macro_halo(node)
                enw = nw + 2.0 * halo
                enh = nh + 2.0 * halo

                # Check that macro fits in slot
                if enw > sw + 1e-6 or enh > sh + 1e-6:
                    valid = False
                    break

                # Center effective macro box in slot, then offset to true macro LL.
                x = sx + (sw - enw) / 2.0 + halo
                y = sy + (sh - enh) / 2.0 + halo
                candidate[node] = (x, y)

            if not valid:
                continue

            hpwl = self._compute_hpwl(candidate)
            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_assignment = dict(candidate)

        if best_assignment is None:
            debug_print(f"{indent}No valid permutation found, falling back to row packing.")
            return self._pack_in_rows(nodes, region)

        debug_print(f"{indent}Best HPWL = {best_hpwl:.1f} for {n} nodes")
        return best_assignment

    def _compute_grid_slots(
        self,
        nodes: List[str],
        region: Region,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Divide a region into a grid of slots for macro placement.

        Returns a list of (x, y, width, height) tuples for each slot.
        """
        n = len(nodes)
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))

        slot_w = region.width / cols
        slot_h = region.height / rows

        slots = []
        for r in range(rows):
            for c in range(cols):
                sx = region.x_min + c * slot_w
                sy = region.y_min + r * slot_h
                slots.append((sx, sy, slot_w, slot_h))

        return slots

    # ──────────────────────────────────────────────────────────
    #  Recursive case: clustering
    # ──────────────────────────────────────────────────────────

    def _cluster_and_recurse(
        self,
        nodes: List[str],
        region: Region,
        depth: int,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Cluster nodes into D groups by connectivity, allocate
        sub-regions proportional to group area, and recurse.
        """
        indent = "  " * depth
        D = min(self.D, len(nodes))

        groups = self._greedy_connectivity_cluster(nodes, D)
        debug_print(f"{indent}Clustered {len(nodes)} nodes into {len(groups)} groups: {[len(g) for g in groups]}")

        # Compute total area per group
        group_areas = []
        for group in groups:
            total = sum(self._node_effective_area(n) for n in group)
            group_areas.append(max(total, 1e-6))  # avoid zero

        # Allocate sub-regions using slicing: alternate horizontal/vertical cuts
        sub_regions = self._allocate_sub_regions(group_areas, region, depth)

        # Recurse on each group
        positions = {}
        for i, group in enumerate(groups):
            sub_positions = self._place_recursive(list(group), sub_regions[i], depth + 1)
            positions.update(sub_positions)

        return positions

    def _greedy_connectivity_cluster(
        self,
        nodes: List[str],
        D: int,
    ) -> List[Set[str]]:
        """
        Greedy agglomerative clustering based on edge connectivity.

        Start with each node in its own cluster, then repeatedly merge
        the two most-connected clusters until D clusters remain.
        Connectivity weight = number of edges between clusters in the
        original graph (reflects bus width / data flow).
        """
        node_set = set(nodes)

        # Initialize: each node is its own cluster
        # cluster_id -> set of nodes
        clusters: Dict[int, Set[str]] = {}
        node_to_cluster: Dict[str, int] = {}
        for i, node in enumerate(nodes):
            clusters[i] = {node}
            node_to_cluster[node] = i

        # Build initial inter-cluster connectivity
        # (cluster_a, cluster_b) -> weight
        connectivity: Dict[Tuple[int, int], int] = {}

        for u, v in self.graph.edges():
            if u not in node_set or v not in node_set:
                continue
            cu = node_to_cluster[u]
            cv = node_to_cluster[v]
            if cu == cv:
                continue
            key = (min(cu, cv), max(cu, cv))
            connectivity[key] = connectivity.get(key, 0) + 1

        # Use a max-heap (negate weights) for efficient merging
        heap = [(-w, a, b) for (a, b), w in connectivity.items()]
        heapq.heapify(heap)

        while len(clusters) > D and heap:
            neg_w, ca, cb = heapq.heappop(heap)

            # Skip if either cluster was already merged
            if ca not in clusters or cb not in clusters:
                continue

            # Merge cb into ca
            for node in clusters[cb]:
                node_to_cluster[node] = ca
            clusters[ca] = clusters[ca] | clusters[cb]
            del clusters[cb]

            # Recompute connectivity for the merged cluster ca
            # against all remaining clusters
            new_conn: Dict[int, int] = {}
            for (a, b), w in list(connectivity.items()):
                # Remove old entries involving ca or cb
                if a == cb or b == cb or a == ca or b == ca:
                    # Determine the "other" cluster
                    other = None
                    if a == ca or a == cb:
                        other = b
                    else:
                        other = a
                    if other == ca or other == cb:
                        continue  # self-edge after merge
                    if other not in clusters:
                        continue
                    new_conn[other] = new_conn.get(other, 0) + w

            # Remove old connectivity entries involving ca or cb
            connectivity = {
                (a, b): w
                for (a, b), w in connectivity.items()
                if a != ca and a != cb and b != ca and b != cb
            }

            # Add new entries for merged cluster
            for other, w in new_conn.items():
                key = (min(ca, other), max(ca, other))
                connectivity[key] = connectivity.get(key, 0) + w
                heapq.heappush(heap, (-connectivity[key], key[0], key[1]))

        return list(clusters.values())

    def _allocate_sub_regions(
        self,
        group_areas: List[float],
        region: Region,
        depth: int,
    ) -> List[Region]:
        """
        Allocate sub-regions within the given region proportional to
        group areas.  Uses a simple slicing approach: at even depths,
        slice horizontally (split width); at odd depths, slice vertically
        (split height).
        """
        total_area = sum(group_areas)
        n = len(group_areas)

        if n == 1:
            return [region]

        # Decide slicing direction based on region shape and depth
        # Prefer cutting the longer dimension
        if region.width >= region.height:
            # Slice along x (vertical cuts → side-by-side sub-regions)
            return self._slice_horizontal(group_areas, region, total_area)
        else:
            # Slice along y (horizontal cuts → stacked sub-regions)
            return self._slice_vertical(group_areas, region, total_area)

    def _slice_horizontal(
        self,
        group_areas: List[float],
        region: Region,
        total_area: float,
    ) -> List[Region]:
        """Split region into vertical strips proportional to group areas."""
        sub_regions = []
        x_cursor = region.x_min
        for i, area in enumerate(group_areas):
            fraction = area / total_area
            strip_width = region.width * fraction
            sub_regions.append(Region(
                x_cursor, region.y_min,
                x_cursor + strip_width, region.y_max
            ))
            x_cursor += strip_width
        return sub_regions

    def _slice_vertical(
        self,
        group_areas: List[float],
        region: Region,
        total_area: float,
    ) -> List[Region]:
        """Split region into horizontal strips proportional to group areas."""
        sub_regions = []
        y_cursor = region.y_min
        for i, area in enumerate(group_areas):
            fraction = area / total_area
            strip_height = region.height * fraction
            sub_regions.append(Region(
                region.x_min, y_cursor,
                region.x_max, y_cursor + strip_height
            ))
            y_cursor += strip_height
        return sub_regions

    # ──────────────────────────────────────────────────────────
    #  HPWL computation
    # ──────────────────────────────────────────────────────────

    def _compute_hpwl(
        self,
        positions: Dict[str, Tuple[float, float]],
    ) -> float:
        """
        Compute total HPWL for the given placement.

        For each edge (u, v) in the graph, compute the half-perimeter
        wire length using pin positions at macro centers: 
            HPWL = |cx_u - cx_v| + |cy_u - cy_v|.

        Nets with multiple sinks are computed using bounding-box HPWL:
            HPWL = (max_x - min_x) + (max_y - min_y)
        over all pins in the net.

        Only edges where both endpoints are in `positions` are counted.
        """
        # Build nets: for each source node, collect all destinations
        # (group by source to handle multi-fanout correctly)
        nets: Dict[str, Set[str]] = {}
        for u, v in self.graph.edges():
            if u in positions and v in positions:
                if u not in nets:
                    nets[u] = set()
                nets[u].add(v)

        total_hpwl = 0.0
        for src, dsts in nets.items():
            # Collect center positions of all pins in this net
            sw, sh = self._node_size[src]
            src_cx = positions[src][0] + sw / 2.0
            src_cy = positions[src][1] + sh / 2.0

            x_coords = [src_cx]
            y_coords = [src_cy]

            for dst in dsts:
                dw, dh = self._node_size[dst]
                dst_cx = positions[dst][0] + dw / 2.0
                dst_cy = positions[dst][1] + dh / 2.0
                x_coords.append(dst_cx)
                y_coords.append(dst_cy)

            net_hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
            total_hpwl += net_hpwl

        return total_hpwl

    # ──────────────────────────────────────────────────────────
    #  Fallback: row packing
    # ──────────────────────────────────────────────────────────

    def _pack_in_rows(
        self,
        nodes: List[str],
        region: Region,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Simple row-based packing as a fallback when enumeration fails.
        Sorts macros by area (large first) and packs left-to-right,
        bottom-to-top.
        """
        # Sort by area descending
        sorted_nodes = sorted(nodes, key=lambda n: self._node_area.get(n, 0), reverse=True)

        positions = {}
        x_cursor = region.x_min + self.macro_gap
        y_cursor = region.y_min + self.macro_gap
        row_max_h = 0.0

        for node in sorted_nodes:
            w, h = self._node_size[node]
            halo = self._small_macro_halo(node)
            ew = w + 2.0 * halo

            # Check if macro fits in current row
            if x_cursor + ew + self.macro_gap > region.x_max:
                # Move to next row
                x_cursor = region.x_min + self.macro_gap
                y_cursor += row_max_h + self.macro_gap
                row_max_h = 0.0

            positions[node] = (x_cursor + halo, y_cursor + halo)
            x_cursor += ew + self.macro_gap
            row_max_h = max(row_max_h, h + 2.0 * halo)

        return positions

    # ──────────────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────────────

    def _snap(self, value: float) -> float:
        """Snap a value to the manufacturing grid."""
        if self.manufacturing_grid <= 0:
            return value
        return round(value / self.manufacturing_grid) * self.manufacturing_grid

    def _small_macro_halo(self, node: str) -> float:
        """
        Dynamic per-side halo around smaller macros.
        Macros at/above median area get no extra halo.
        """
        if not self.enable_small_macro_halo:
            return 0.0
        area = self._node_area.get(node, 0.0)
        if area <= 0.0 or self._median_macro_area <= 0.0:
            return 0.0
        threshold = self._median_macro_area * self.small_macro_halo_median_factor
        if area >= threshold:
            return 0.0
        ratio = (threshold / area) - 1.0
        halo = self.macro_gap * ratio
        return max(0.0, min(halo, self.max_small_macro_halo))

    def _node_effective_area(self, node: str) -> float:
        """Area including dynamic halo, used for region allocation."""
        w, h = self._node_size.get(node, (0.0, 0.0))
        halo = self._small_macro_halo(node)
        return (w + 2.0 * halo) * (h + 2.0 * halo)
