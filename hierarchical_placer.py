"""
Hierarchical Macro Placer

Implements a recursive placement algorithm for macro blocks:
  - If the number of macros exceeds threshold N, use connectivity- and area-aware
    clustering to partition into exactly min(D, n) groups, then recurse on each group.
  - If the number of macros is <= N, enumerate all permutations of macros into
    D grid slots and pick the arrangement with minimum HPWL.

The placer operates on the functional-unit-level graph (one node per Add/Mult/Mux/etc.)
and produces (x, y) coordinates for each macro, which are written to a TCL script
that can be sourced by OpenROAD in place of rtl_macro_placer.
"""

import logging
import math
import itertools
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

## Number of groups for each recursive partition (and preferred slot columns in enumeration)
DEFAULT_D = 4
## Scales the area-imbalance penalty in merge scoring: score = edge_count - weight * penalty.
## Tuning: larger values favor more equal macro-area per cluster (avoid one huge cluster with
## many tiny siblings); merges that overshoot the per-group area target are penalized more, so
## connectivity can be sacrificed for balance. Smaller values favor preserving dense dataflow
## (merge highly connected clusters even if areas become skewed); use toward 0 for
## connectivity-dominated clustering.
DEFAULT_CLUSTER_AREA_BALANCE_WEIGHT = 10.0

## Aspect ratio bounds for sub-regions (height / width)
DEFAULT_MIN_ASPECT_RATIO = 0.33
DEFAULT_MAX_ASPECT_RATIO = 3.0

## Gap between macros (microns) – used as minimum spacing
DEFAULT_MACRO_GAP = 10.0
DEFAULT_ENABLE_SMALL_MACRO_HALO = True
DEFAULT_MAX_SMALL_MACRO_HALO = 20.0
# LEF macro names; halo applies only when node_to_macro_name matches (case-insensitive).
DEFAULT_SMALL_MACRO_HALO_MACRO_NAMES = frozenset(
    {"Mux16", "And16", "Not16", "Or16", "BitAnd16", "BitOr16", "BitXor16", "Eq16", "Gt16", "GtE16", "Lt16", "LtE16", "NotEq16", "Register16", "Add16", "LShift16", "RShift16", "Sub16"}
)
## Inset applied to core on all sides for the root placement region (reduces FP / snap pushing past core).
DEFAULT_PLACEMENT_INSET_UM = 5.0
## When die_area is supplied, macro LEF bbox is clamped inside die minus this margin (per side).
DEFAULT_DIE_INNER_MARGIN_UM = 2.0


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
        Number of child groups at every recursive split (always merged down
        to exactly ``min(D, len(nodes))`` clusters). Also used as the number
        of slot columns in the enumeration grid (rows are sized to fit all nodes).
    cluster_area_balance_weight : float
        Multiplier on the squared area overshoot penalty in each merge decision
        (``edge_count - weight * penalty``). **Larger** values push partitions toward
        similar total macro area per cluster, at the cost of sometimes merging fewer
        highly connected neighbors. **Smaller** values keep connectivity primary so
        tightly coupled macros merge even when that yields uneven cluster areas.
    min_aspect_ratio : float
        Minimum allowed aspect ratio (height / width) for sub-regions.
    max_aspect_ratio : float
        Maximum allowed aspect ratio (height / width) for sub-regions.
    macro_gap : float
        Minimum gap between placed macros (microns).
    max_small_macro_halo : float
        Per-side halo in microns for whitelisted macro names only.
    small_macro_halo_macro_names : Optional[Set[str]]
        Only these LEF macro names receive the halo (compared case-insensitively).
        None uses DEFAULT_SMALL_MACRO_HALO_MACRO_NAMES.
    manufacturing_grid : float
        Manufacturing grid for snapping coordinates (microns).
        Defaults to 0.005 (FreePDK45).
    die_area : tuple (x_min, y_min, x_max, y_max) | None
        Optional die rectangle in microns (same convention as DEF DIEAREA). When set,
        each macro's LEF bbox is clamped inside the die after placement so global
        routing does not see pins outside the die. OpenROAD die is typically larger
        than core; passing this matches GRT's boundary check.
    placement_inset_um : float
        Shrinks the **core** rectangle on all sides before the top-level recursive
        place (only affects the root region; sub-regions stay inside it).
    die_inner_margin_um : float
        Extra inset from die edges when applying die clamping.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        macro_size_dict: Dict[str, Tuple[float, float]],
        node_to_macro_name: Dict[str, str],
        core_area: Tuple[float, float, float, float],
        N: int = DEFAULT_N,
        D: int = DEFAULT_D,
        cluster_area_balance_weight: float = DEFAULT_CLUSTER_AREA_BALANCE_WEIGHT,
        min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO,
        max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
        macro_gap: float = DEFAULT_MACRO_GAP,
        enable_small_macro_halo: bool = DEFAULT_ENABLE_SMALL_MACRO_HALO,
        max_small_macro_halo: float = DEFAULT_MAX_SMALL_MACRO_HALO,
        small_macro_halo_macro_names: Optional[Set[str]] = None,
        die_area: Optional[Tuple[float, float, float, float]] = None,
        placement_inset_um: float = DEFAULT_PLACEMENT_INSET_UM,
        die_inner_margin_um: float = DEFAULT_DIE_INNER_MARGIN_UM,
        manufacturing_grid: float = 0.005,
    ):
        self.graph = graph
        self.macro_size_dict = macro_size_dict
        self.node_to_macro_name = node_to_macro_name
        self.core_region = Region(*core_area)
        self.die_inner_margin_um = die_inner_margin_um
        self._die_region: Optional[Region] = Region(*die_area) if die_area is not None else None

        inset = placement_inset_um
        cr = self.core_region
        self._floorplan_region = Region(
            cr.x_min + inset, cr.y_min + inset, cr.x_max - inset, cr.y_max - inset
        )
        if self._floorplan_region.width <= 0.0 or self._floorplan_region.height <= 0.0:
            debug_print(
                f"placement_inset_um={inset} leaves non-positive floorplan region; using core as-is."
            )
            self._floorplan_region = cr
        self.N = N
        self.D = D
        self.cluster_area_balance_weight = cluster_area_balance_weight
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.macro_gap = macro_gap
        self.enable_small_macro_halo = enable_small_macro_halo
        self.max_small_macro_halo = max_small_macro_halo
        names = (
            small_macro_halo_macro_names
            if small_macro_halo_macro_names is not None
            else DEFAULT_SMALL_MACRO_HALO_MACRO_NAMES
        )
        self._small_macro_halo_macro_names_upper = frozenset(
            n.upper() for n in names
        )
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

        debug_print(
            f"Placing {len(nodes)} macros: core={self.core_region}, "
            f"floorplan={self._floorplan_region}, die={self._die_region}"
        )
        positions = self._place_recursive(nodes, self._floorplan_region, depth=0)

        snapped: Dict[str, Tuple[float, float]] = {}
        for node, (x, y) in positions.items():
            x, y = self._snap(x), self._snap(y)
            w, h = self._node_size[node]
            x, y = self._clamp_macro_bbox_ll(
                x, y, w, h, self._floorplan_region.x_min, self._floorplan_region.y_min,
                self._floorplan_region.x_max, self._floorplan_region.y_max,
            )
            if self._die_region is not None:
                m = self.die_inner_margin_um
                dr = self._die_region
                x, y = self._clamp_macro_bbox_ll(
                    x, y, w, h, dr.x_min + m, dr.y_min + m, dr.x_max - m, dr.y_max - m,
                )
            snapped[node] = (x, y)

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
        # Prefer a fixed column count of D (same fan-out as recursive clustering), not sqrt(n).
        cols = max(1, min(self.D, n))
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
        Cluster nodes into exactly D_eff groups (D_eff = min(self.D, len(nodes))) using
        connectivity and area balance, allocate area-proportional strips (orientation
        alternates with depth: horizontal strips at even depth, vertical at odd), then recurse.
        """
        indent = "  " * depth
        D_eff = min(self.D, len(nodes))

        groups = self._area_aware_connectivity_cluster(nodes, D_eff)
        assert len(groups) == D_eff, f"expected {D_eff} groups, got {len(groups)}"
        debug_print(
            f"{indent}Clustered {len(nodes)} nodes into {len(groups)} groups (D={self.D}): "
            f"{[len(g) for g in groups]}, areas={[round(sum(self._node_effective_area(n) for n in g), 2) for g in groups]}"
        )

        # Compute total area per group
        group_areas = []
        for group in groups:
            total = sum(self._node_effective_area(n) for n in group)
            group_areas.append(max(total, 1e-6))  # avoid zero

        # Allocate sub-regions as strips; orientation alternates by recursion depth
        sub_regions = self._allocate_sub_regions(group_areas, region, depth)

        # Recurse on each group
        positions = {}
        for i, group in enumerate(groups):
            sub_positions = self._place_recursive(list(group), sub_regions[i], depth + 1)
            positions.update(sub_positions)

        return positions

    def _area_aware_connectivity_cluster(
        self,
        nodes: List[str],
        D: int,
    ) -> List[Set[str]]:
        """
        Agglomerative clustering until exactly D clusters remain.

        Each merge scores connectivity (inter-cluster edge count) minus a
        quadratic penalty when merged effective area exceeds total/D, so
        partitions stay closer to equal macro area while preserving dataflow.

        Always performs len(nodes) - D merges (unlike a pure connectivity heap,
        which can stop early on sparse graphs and leave more than D clusters).
        """
        node_set = set(nodes)

        clusters: Dict[int, Set[str]] = {}
        node_to_cluster: Dict[str, int] = {}
        for i, node in enumerate(nodes):
            clusters[i] = {node}
            node_to_cluster[node] = i

        cluster_area: Dict[int, float] = {
            cid: sum(self._node_effective_area(n) for n in group)
            for cid, group in clusters.items()
        }
        total_area = sum(cluster_area.values())
        target = total_area / max(D, 1)

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

        def merge_score(ca: int, cb: int) -> float:
            # See DEFAULT_CLUSTER_AREA_BALANCE_WEIGHT comment: higher weight -> stronger area balance.
            conn_w = connectivity.get((min(ca, cb), max(ca, cb)), 0)
            merged_area = cluster_area[ca] + cluster_area[cb]
            over = max(0.0, merged_area - target)
            penalty = (over / max(target, 1e-9)) ** 2
            return float(conn_w) - self.cluster_area_balance_weight * penalty

        def pick_best_pair() -> Tuple[int, int]:
            ids = sorted(clusters.keys())
            best_pair: Optional[Tuple[int, int]] = None
            best_key: Optional[Tuple[float, int, float, float]] = None
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    ca, cb = ids[i], ids[j]
                    conn_w = connectivity.get((min(ca, cb), max(ca, cb)), 0)
                    sc = merge_score(ca, cb)
                    merged_area = cluster_area[ca] + cluster_area[cb]
                    mx = max(cluster_area[ca], cluster_area[cb])
                    err = abs(merged_area - target)
                    key = (sc, conn_w, -mx, -err)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_pair = (ca, cb)
            assert best_pair is not None
            return best_pair

        while len(clusters) > D:
            ca, cb = pick_best_pair()

            for node in clusters[cb]:
                node_to_cluster[node] = ca
            clusters[ca] = clusters[ca] | clusters[cb]
            del clusters[cb]
            cluster_area[ca] = cluster_area[ca] + cluster_area[cb]
            del cluster_area[cb]

            new_conn: Dict[int, int] = {}
            for (a, b), w in list(connectivity.items()):
                if a == cb or b == cb or a == ca or b == ca:
                    other = None
                    if a == ca or a == cb:
                        other = b
                    else:
                        other = a
                    if other == ca or other == cb:
                        continue
                    if other not in clusters:
                        continue
                    new_conn[other] = new_conn.get(other, 0) + w

            connectivity = {
                (a, b): w
                for (a, b), w in connectivity.items()
                if a != ca and a != cb and b != ca and b != cb
            }

            for other, w in new_conn.items():
                key = (min(ca, other), max(ca, other))
                connectivity[key] = connectivity.get(key, 0) + w

        return list(clusters.values())

    def _allocate_sub_regions(
        self,
        group_areas: List[float],
        region: Region,
        depth: int,
    ) -> List[Region]:
        """
        Allocate sub-regions as strips whose widths (or heights) are proportional
        to group macro area.

        Strip orientation alternates with recursion depth (``_place_recursive`` /
        ``_cluster_and_recurse`` depth): **even** depth uses **horizontal** strips
        (stacked rows, full parent width — implemented as ``_slice_vertical``),
        **odd** depth uses **vertical** strips (side-by-side columns, full parent
        height — ``_slice_horizontal``).

        Each slice's rectangle area is proportional to the sum of effective macro
        areas in that cluster (``sum(_node_effective_area)`` for nodes in the group).
        """
        n = len(group_areas)

        if n == 1:
            return [region]

        # depth 0 = top-level partition after initial clustering
        if depth % 2 == 0:
            return self._slice_vertical(group_areas, region)
        return self._slice_horizontal(group_areas, region)

    def _slice_horizontal(
        self,
        group_areas: List[float],
        region: Region,
    ) -> List[Region]:
        """
        Vertical strips (columns): each slice spans the full parent height, slice
        width is ``region.width * (macro_area_i / sum(macro_areas))``. Slice
        geometric area is therefore ``region.area * macro_area_i / sum(...)`` —
        proportional to the macro area budget for that group.
        """
        T = sum(group_areas)
        if T <= 0.0:
            T = 1e-9
        W = region.width
        H = region.height
        x0 = region.x_min
        y0, y1 = region.y_min, region.y_max
        x1_bound = region.x_max
        sub_regions: List[Region] = []
        n = len(group_areas)
        for i, a in enumerate(group_areas):
            if i == n - 1:
                x1 = x1_bound
            else:
                x1 = x0 + W * (a / T)
            sub_regions.append(Region(x0, y0, x1, y1))
            x0 = x1
        return sub_regions

    def _slice_vertical(
        self,
        group_areas: List[float],
        region: Region,
    ) -> List[Region]:
        """
        Horizontal strips (rows): each slice spans the full parent width, slice
        height is ``region.height * (macro_area_i / sum(macro_areas))``. Slice
        geometric area is ``region.area * macro_area_i / sum(...)``.
        """
        T = sum(group_areas)
        if T <= 0.0:
            T = 1e-9
        W = region.width
        H = region.height
        x0, x1 = region.x_min, region.x_max
        y0 = region.y_min
        y1_bound = region.y_max
        sub_regions: List[Region] = []
        n = len(group_areas)
        for i, a in enumerate(group_areas):
            if i == n - 1:
                y1 = y1_bound
            else:
                y1 = y0 + H * (a / T)
            sub_regions.append(Region(x0, y0, x1, y1))
            y0 = y1
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

        # Row packing does not prove y extent fits; clamp LL so LEF bbox stays in region.
        for node in list(positions):
            w, h = self._node_size[node]
            x, y = positions[node]
            nx, ny = self._clamp_macro_bbox_ll(
                x, y, w, h,
                region.x_min, region.y_min, region.x_max, region.y_max,
            )
            if (nx, ny) != (x, y):
                debug_print(
                    f"_pack_in_rows: clamped {node} from ({x:.3f},{y:.3f}) to ({nx:.3f},{ny:.3f}) "
                    f"to fit {region}"
                )
            positions[node] = (nx, ny)

        return positions

    # ──────────────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────────────

    def _clamp_macro_bbox_ll(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> Tuple[float, float]:
        """
        Clamp macro lower-left so the LEF SIZE box [x,x+w]×[y,y+h] lies in
        [xmin,xmax]×[ymin,ymax] (x/y are edges; macro must satisfy x+w<=xmax, y+h<=ymax).
        """
        if w <= 0.0 or h <= 0.0:
            return x, y
        if xmax - xmin < w - 1e-9 or ymax - ymin < h - 1e-9:
            debug_print(
                f"_clamp_macro_bbox_ll: macro {w}x{h} um does not fit in "
                f"({xmin},{ymin})-({xmax},{ymax}); pinning LL to corner."
            )
            return xmin, ymin
        x = max(xmin, min(x, xmax - w))
        y = max(ymin, min(y, ymax - h))
        return x, y

    def _snap(self, value: float) -> float:
        """Snap a value to the manufacturing grid."""
        if self.manufacturing_grid <= 0:
            return value
        return round(value / self.manufacturing_grid) * self.manufacturing_grid

    def _small_macro_halo(self, node: str) -> float:
        """Fixed per-side halo for whitelisted LEF macro names only."""
        if not self.enable_small_macro_halo:
            return 0.0
        macro_name = self.node_to_macro_name.get(node)
        if not macro_name or macro_name.upper() not in self._small_macro_halo_macro_names_upper:
            return 0.0
        return max(0.0, self.max_small_macro_halo)

    def _node_effective_area(self, node: str) -> float:
        """Area including name-based halo, used for region allocation."""
        w, h = self._node_size.get(node, (0.0, 0.0))
        halo = self._small_macro_halo(node)
        return (w + 2.0 * halo) * (h + 2.0 * halo)
