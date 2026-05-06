"""
Hierarchical Macro Placer

Implements a recursive placement algorithm for macro blocks:
  - If the number of macros exceeds threshold N, use connectivity- and area-aware
    clustering to partition into exactly min(D, n) groups, then recurse on each group.
  - If the number of macros is <= N, enumerate permutations into a **slack-aware**
    grid (prefers slot dimensions that comfortably fit the largest macro envelope),
    then HPWL. When enumeration is infeasible or too large, try overlap-safe grid
    packing **before** a **limited** set of alternate (cols, rows) shapes (feasible
    only, capped). Final pass runs stronger bbox separation + snap.

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
DEFAULT_MIN_ASPECT_RATIO = 0.25
DEFAULT_MAX_ASPECT_RATIO = 4.0

## Gap between macros (microns) – used as minimum spacing
DEFAULT_MACRO_GAP = 10.0
## Max (cols, rows) shapes to try in alternate-grid enumeration (after feasibility filter).
DEFAULT_MAX_ALTERNATE_GRID_SHAPES = 12
# Deprecated halo knobs (kept for config compatibility).
# `max_small_macro_halo` is repurposed as a uniform internal placement padding term
# to preserve legacy clustering/partitioning behavior after removing physical halos.
DEFAULT_ENABLE_SMALL_MACRO_HALO = False
DEFAULT_MAX_SMALL_MACRO_HALO = 10.0
DEFAULT_SMALL_MACRO_HALO_MACRO_NAMES = frozenset()
## Inset applied to core on all sides for the root placement region (reduces FP / snap pushing past core).
DEFAULT_PLACEMENT_INSET_UM = 5.0
## When die_area is supplied, macro LEF bbox is clamped inside die minus this margin (per side).
DEFAULT_DIE_INNER_MARGIN_UM = 2.0
## Sum of macro LEF areas (µm²) × this factor → target area for a centered placement window
## inside the floorplan (same aspect as floorplan). Improves HPWL by concentrating macros.
## Set <= 0 to disable and use the full floorplan region.
DEFAULT_PLACEMENT_ROUTING_AREA_SCALE = 4.0


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
    enable_small_macro_halo / small_macro_halo_macro_names :
        Deprecated and ignored. Kept only for backward compatibility with older configs.
    max_small_macro_halo :
        Deprecated name, but still used as a **uniform internal placement padding** (µm, per side)
        for clustering/partitioning/slot feasibility. No physical halo is applied to macro origins.
    manufacturing_grid : float
        Manufacturing grid for snapping coordinates (microns).
        Defaults to 0.005 (FreePDK45).
    dbu_per_micron : int
        DEF/LEF database units per micron (e.g. 2000 for FreePDK45). Used with
        ``manufacturing_grid`` so snapped origins are multiples of the grid in **DBU**,
        matching DRT pin-shape alignment.
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
    placement_routing_area_scale : float
        Multiplier on the sum of macro LEF areas to set the area of a centered placement
        window (subset of the floorplan). See ``DEFAULT_PLACEMENT_ROUTING_AREA_SCALE``.
        If <= 0, the full floorplan is used.
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
        placement_routing_area_scale: float = DEFAULT_PLACEMENT_ROUTING_AREA_SCALE,
        dbu_per_micron: int = 2000,
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
        # Halo feature is removed physically, but we keep a uniform internal padding term
        # to preserve legacy placement logic quality.
        self._internal_layout_padding_um = max(0.0, float(max_small_macro_halo))
        self.manufacturing_grid = manufacturing_grid
        self.placement_routing_area_scale = placement_routing_area_scale
        self.dbu_per_micron = max(1, int(dbu_per_micron))
        self._active_placement_region = self._floorplan_region
        self.max_alternate_grid_shapes = DEFAULT_MAX_ALTERNATE_GRID_SHAPES

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

    def _compute_centered_placement_subset(self, nodes: List[str]) -> Region:
        """
        Rectangle centered on the floorplan, matching its aspect ratio, whose area is
        ``min(floorplan_area, max(sum_macro_lef_areas * scale, sum_macro_lef_areas))``.

        If ``placement_routing_area_scale`` <= 0 or the target area is not below the
        floorplan area, returns the full floorplan region.
        """
        fp = self._floorplan_region
        w_fp = fp.width
        h_fp = fp.height
        if w_fp <= 0.0 or h_fp <= 0.0:
            return fp
        scale = self.placement_routing_area_scale
        if scale <= 0.0:
            return fp
        a_macro = sum(self._node_area.get(n, 0.0) for n in nodes)
        if a_macro <= 0.0:
            return fp
        a_need = max(a_macro * scale, a_macro)
        a_fp = w_fp * h_fp
        if a_need >= a_fp - 1e-9:
            return fp
        aspect = w_fp / h_fp
        w_sub = math.sqrt(a_need * aspect)
        h_sub = math.sqrt(a_need / aspect)
        if w_sub > w_fp + 1e-9:
            s = w_fp / w_sub
            w_sub *= s
            h_sub *= s
        if h_sub > h_fp + 1e-9:
            s = h_fp / h_sub
            w_sub *= s
            h_sub *= s
        cx, cy = fp.cx, fp.cy
        return Region(
            cx - 0.5 * w_sub,
            cy - 0.5 * h_sub,
            cx + 0.5 * w_sub,
            cy + 0.5 * h_sub,
        )

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

        self._active_placement_region = self._compute_centered_placement_subset(nodes)
        ar = self._active_placement_region
        debug_print(
            f"Placing {len(nodes)} macros: core={self.core_region}, "
            f"floorplan={self._floorplan_region}, active_window={ar}, die={self._die_region}"
        )
        positions = self._place_recursive(nodes, ar, depth=0)

        snapped: Dict[str, Tuple[float, float]] = {}
        for node, (x, y) in positions.items():
            x, y = self._snap(x), self._snap(y)
            w, h = self._node_size[node]
            x, y = self._clamp_macro_bbox_ll(
                x, y, w, h, ar.x_min, ar.y_min, ar.x_max, ar.y_max,
            )
            if self._die_region is not None:
                m = self.die_inner_margin_um
                dr = self._die_region
                x, y = self._clamp_macro_bbox_ll(
                    x, y, w, h, dr.x_min + m, dr.y_min + m, dr.x_max - m, dr.y_max - m,
                )
            snapped[node] = (x, y)

        snapped = self._legalize_global_overlaps(snapped)
        snapped = self._snap_all_macro_positions(snapped)
        g = self.macro_gap
        for _ in range(12):
            if not self._layout_has_bbox_violations(snapped, g - 1e-9):
                break
            snapped = self._legalize_push_apart(
                snapped, self._active_placement_region, clearance=g, max_iters=768
            )
            snapped = self._snap_all_macro_positions(snapped)
        if self._layout_has_bbox_violations(snapped, g - 1e-9):
            debug_print(
                "Warning: could not fully separate all macro bboxes within the active placement window; "
                "check core size, macro_gap, placement_routing_area_scale, or hierarchy depth."
            )

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

                x_dbu = self._def_dbu_ll_from_micron(x)
                y_dbu = self._def_dbu_ll_from_micron(y)

                f.write(f'# Node: {node}\n')
                f.write(f'set _hp_inst [$_hp_block findInst "{component_id}"]\n')
                f.write(f'if {{$_hp_inst != "NULL"}} {{\n')
                f.write(f"  set _hp_x {x_dbu}\n")
                f.write(f"  set _hp_y {y_dbu}\n")
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
            # Trivial: center the single macro in the region (layout pad for bounds; LEF w×h)
            node = nodes[0]
            w, h = self._node_size[node]
            m = self._layout_expansion_half_um(node)
            ew = w + 2.0 * m
            eh = h + 2.0 * m
            x = region.cx - ew / 2.0 + m
            y = region.cy - eh / 2.0 + m
            x = max(region.x_min + m, min(x, region.x_max - w - m))
            y = max(region.y_min + m, min(y, region.y_max - h - m))
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
            debug_print(
                f"{indent}Warning: only {len(slots)} slots for {n} nodes; using non-overlapping grid pack."
            )
            return self._pack_uniform_grid_no_overlap(nodes, region)

        # Enumerate all permutations of slot assignments
        best_hpwl = float("inf")
        best_assignment = None

        # For efficiency, limit permutation count.  If n > 8, this would be 8! = 40320 which
        # is already borderline. The N parameter should keep this in check.
        slot_indices = list(range(len(slots)))
        max_perm = 200000
        p_count = 1
        for i in range(n):
            p_count *= len(slots) - i
        if p_count > max_perm:
            debug_print(
                f"{indent}Slot assignment count {p_count} too large; "
                f"trying guaranteed grid pack before alternate grids."
            )
            packed = self._pack_uniform_grid_no_overlap(nodes, region)
            if not self._layout_has_bbox_violations(packed, self.macro_gap - 1e-9):
                return packed
            alt = self._enumerate_placements_alternate_grids(nodes, region, depth)
            if alt is not None:
                return alt
            return packed

        for perm in itertools.permutations(slot_indices, n):
            # Assign each node to a slot
            candidate = {}
            valid = True
            for i, node in enumerate(nodes):
                slot_idx = perm[i]
                sx, sy, sw, sh = slots[slot_idx]
                nw, nh = self._node_size[node]
                ew, eh = self._effective_macro_envelope(node)
                if ew > sw + 1e-6 or eh > sh + 1e-6:
                    valid = False
                    break
                x = sx + (sw - nw) / 2.0
                y = sy + (sh - nh) / 2.0
                candidate[node] = (x, y)

            if not valid:
                continue

            hpwl = self._compute_hpwl(candidate)
            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_assignment = dict(candidate)

        if best_assignment is None:
            debug_print(
                f"{indent}No valid permutation on default grid; "
                f"trying non-overlapping pack before a limited alternate-grid search."
            )
            packed = self._pack_uniform_grid_no_overlap(nodes, region)
            if not self._layout_has_bbox_violations(packed, self.macro_gap - 1e-9):
                return packed
            alt = self._enumerate_placements_alternate_grids(nodes, region, depth)
            if alt is not None:
                return alt
            return packed

        debug_print(f"{indent}Best HPWL = {best_hpwl:.1f} for {n} nodes")
        return best_assignment

    def _effective_macro_envelope(self, node: str) -> Tuple[float, float]:
        """Outer width/height for slot and strip feasibility (layout pad, not extra LEF metal)."""
        nw, nh = self._node_size[node]
        m = self._layout_expansion_half_um(node)
        return (nw + 2.0 * m, nh + 2.0 * m)

    def _gapped_slot_dimensions(
        self,
        region: Region,
        cols: int,
        rows: int,
    ) -> Tuple[float, float]:
        """Usable slot width/height with ``macro_gap`` between slots and at region edges."""
        gap = self.macro_gap
        cols = max(1, cols)
        rows = max(1, rows)
        aw = max(0.0, region.width - (cols + 1) * gap)
        ah = max(0.0, region.height - (rows + 1) * gap)
        return (aw / cols, ah / rows)

    def _grid_slots_for_shape(
        self,
        region: Region,
        cols: int,
        rows: int,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Grid of slots separated by ``macro_gap`` so adjacent placed macros keep
        minimum spacing even when the small-macro halo is disabled.
        """
        gap = self.macro_gap
        cols = max(1, cols)
        rows = max(1, rows)
        slot_w, slot_h = self._gapped_slot_dimensions(region, cols, rows)
        slots: List[Tuple[float, float, float, float]] = []
        for r in range(rows):
            for c in range(cols):
                sx = region.x_min + gap + c * (slot_w + gap)
                sy = region.y_min + gap + r * (slot_h + gap)
                slots.append((sx, sy, slot_w, slot_h))
        return slots

    def _enumerate_on_slots(
        self,
        nodes: List[str],
        region: Region,
        slots: List[Tuple[float, float, float, float]],
        depth: int,
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """HPWL-best placement over assignments of n nodes to n distinct slots."""
        n = len(nodes)
        if len(slots) < n:
            return None
        max_perm = 200000
        slot_indices = list(range(len(slots)))
        p_len = 1
        for i in range(n):
            p_len *= len(slots) - i
        if p_len > max_perm:
            return None

        indent = "  " * depth
        best_hpwl = float("inf")
        best_assignment: Optional[Dict[str, Tuple[float, float]]] = None

        for perm in itertools.permutations(slot_indices, n):
            candidate: Dict[str, Tuple[float, float]] = {}
            valid = True
            for i, node in enumerate(nodes):
                slot_idx = perm[i]
                sx, sy, sw, sh = slots[slot_idx]
                nw, nh = self._node_size[node]
                ew, eh = self._effective_macro_envelope(node)
                if ew > sw + 1e-6 or eh > sh + 1e-6:
                    valid = False
                    break
                x = sx + (sw - nw) / 2.0
                y = sy + (sh - nh) / 2.0
                candidate[node] = (x, y)
            if not valid:
                continue
            hpwl = self._compute_hpwl(candidate)
            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_assignment = dict(candidate)

        if best_assignment is None:
            return None
        debug_print(f"{indent}alternate grid: Best HPWL = {best_hpwl:.1f} for {n} nodes")
        return best_assignment

    def _enumerate_placements_alternate_grids(
        self,
        nodes: List[str],
        region: Region,
        depth: int,
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Try several (cols, rows) shapes so tall/wide regions still get feasible slots
        without resorting to shelf packing.
        """
        n = len(nodes)
        max_ew = max(self._effective_macro_envelope(node)[0] for node in nodes)
        max_eh = max(self._effective_macro_envelope(node)[1] for node in nodes)
        best: Optional[Dict[str, Tuple[float, float]]] = None
        best_hpwl = float("inf")
        seen: Set[Tuple[int, int]] = set()
        candidates: List[Tuple[int, int]] = []

        for cols in range(1, n + 1):
            rows = max(1, math.ceil(n / cols))
            if cols * rows < n:
                continue
            key = (cols, rows)
            if key not in seen:
                seen.add(key)
                candidates.append(key)
        for rows in range(1, n + 1):
            cols = max(1, math.ceil(n / rows))
            if cols * rows < n:
                continue
            key = (cols, rows)
            if key not in seen:
                seen.add(key)
                candidates.append(key)

        ranked: List[Tuple[float, int, int, int]] = []
        for cols, rows in candidates:
            sw, sh = self._gapped_slot_dimensions(region, cols, rows)
            if sw + 1e-9 < max_ew or sh + 1e-9 < max_eh:
                continue
            slack = min(sw - max_ew, sh - max_eh)
            ranked.append((-slack, cols * rows, cols, rows))
        ranked.sort()
        cap = max(4, min(self.max_alternate_grid_shapes, max(8, 2 * self.D + 4)))
        for _neg_slack, _cells, cols, rows in ranked[:cap]:
            slots = self._grid_slots_for_shape(region, cols, rows)
            got = self._enumerate_on_slots(nodes, region, slots, depth)
            if got is None:
                continue
            h = self._compute_hpwl(got)
            if h < best_hpwl:
                best_hpwl = h
                best = got
        return best

    def _compute_grid_slots(
        self,
        nodes: List[str],
        region: Region,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Divide a region into a grid of slots for macro placement.

        Returns a list of (x, y, width, height) tuples for each slot.
        Picks (cols, rows) among simple candidates so the smallest slot still fits
        the largest macro envelope when possible (reduces dead enumeration / fallbacks).
        """
        n = len(nodes)
        if n == 0:
            return []

        max_ew = max(self._effective_macro_envelope(node)[0] for node in nodes)
        max_eh = max(self._effective_macro_envelope(node)[1] for node in nodes)

        def score_shape(cols: int, rows: int) -> Tuple[int, float, float]:
            """Prefer feasible grids with the most slot slack (room for macros), then larger min side."""
            cols = max(1, cols)
            rows = max(1, rows)
            if cols * rows < n:
                return (-1, float("-inf"), -1.0)
            sw, sh = self._gapped_slot_dimensions(region, cols, rows)
            min_side = min(sw, sh)
            margin_w = sw - max_ew
            margin_h = sh - max_eh
            fits = 1 if margin_w >= -1e-6 and margin_h >= -1e-6 else 0
            slack = min(margin_w, margin_h)
            return (fits, slack, min_side)

        best_key: Optional[Tuple[int, float, float]] = None
        best_shape: Tuple[int, int] = (max(1, min(self.D, n)), max(1, math.ceil(n / max(1, min(self.D, n)))))

        seen: Set[Tuple[int, int]] = set()
        for cols in range(1, n + 1):
            rows = max(1, math.ceil(n / cols))
            if cols * rows < n:
                continue
            key = (cols, rows)
            if key in seen:
                continue
            seen.add(key)
            k = score_shape(cols, rows)
            if best_key is None or k > best_key:
                best_key = k
                best_shape = (cols, rows)

        cols, rows = best_shape
        return self._grid_slots_for_shape(region, cols, rows)

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
        sub_regions = self._allocate_sub_regions(group_areas, region, depth, groups)
        # If any subgroup region cannot even contain the largest macro envelope in that
        # subgroup, recursive containment clamps will pin multiple macros to the same
        # corner and create overlaps. Fall back to one-level non-overlap pack here.
        for i, group in enumerate(groups):
            child = sub_regions[i]
            need_w = self._min_packed_width_in_group(group)
            need_h = self._min_packed_height_in_group(group)
            if child.width + 1e-9 < need_w or child.height + 1e-9 < need_h:
                debug_print(
                    f"{indent}Subregion infeasible for group {i}: child=({child.width:.3f}x{child.height:.3f}) "
                    f"need>=({need_w:.3f}x{need_h:.3f}); falling back to parent-level non-overlap pack."
                )
                return self._pack_uniform_grid_no_overlap(nodes, region)

        # Recurse on each group
        positions = {}
        for i, group in enumerate(groups):
            sub_positions = self._place_recursive(list(group), sub_regions[i], depth + 1)
            # Hard containment at each recursion level: keep full macro LEF bbox
            # inside the assigned child region.
            child = sub_regions[i]
            for node, (x, y) in sub_positions.items():
                w, h = self._node_size[node]
                x, y = self._clamp_macro_bbox_ll(
                    x, y, w, h, child.x_min, child.y_min, child.x_max, child.y_max
                )
                # Re-snap after containment clamp to keep LLs on manufacturing grid.
                x, y = self._snap_macro_ll_in_region(node, x, y, child)
                sub_positions[node] = (x, y)
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
        groups: List[Set[str]],
    ) -> List[Region]:
        """
        Allocate sub-regions as strips with ``macro_gap`` gutters between siblings
        and at least enough width/height for the largest packed footprint in each group.

        Strip orientation alternates with recursion depth: **even** depth uses
        horizontal strips (stacked rows, full parent width), **odd** depth uses
        vertical strips (columns, full parent height).
        """
        n = len(group_areas)

        if n == 1:
            return [region]

        if depth % 2 == 0:
            return self._stack_horizontal_stripes(groups, group_areas, region)
        return self._stack_vertical_stripes(groups, group_areas, region)

    def _min_packed_width_in_group(self, group: Set[str]) -> float:
        return max(
            (self._node_size[n][0] + 2.0 * self._layout_expansion_half_um(n) for n in group),
            default=0.0,
        )

    def _min_packed_height_in_group(self, group: Set[str]) -> float:
        return max(
            (self._node_size[n][1] + 2.0 * self._layout_expansion_half_um(n) for n in group),
            default=0.0,
        )

    def _stack_horizontal_stripes(
        self,
        groups: List[Set[str]],
        group_areas: List[float],
        region: Region,
    ) -> List[Region]:
        """
        Full-width rows separated by ``macro_gap``; each row height is at least
        the tallest packed footprint in that cluster, with extra height split by area weights.
        """
        n = len(groups)
        gap = self.macro_gap
        T = sum(group_areas)
        if T <= 0.0:
            T = 1e-9
        mins = [self._min_packed_height_in_group(g) for g in groups]
        gaps_total = (n - 1) * gap
        H_eff = max(0.0, region.height - gaps_total)
        slack = H_eff - sum(mins)
        if slack < -1e-6:
            debug_print(
                f"_stack_horizontal_stripes: region height {region.height:.3f} um cannot "
                f"fit min row heights {sum(mins):.3f} um plus {gaps_total:.3f} um gutters; "
                "using area-only split (placement may be tight)."
            )
            heights = [H_eff * (group_areas[i] / T) for i in range(n)]
        else:
            heights = [
                mins[i] + max(0.0, slack) * (group_areas[i] / T) for i in range(n)
            ]
        y0 = region.y_min
        sub_regions: List[Region] = []
        for i in range(n):
            if i == n - 1:
                sub_regions.append(
                    Region(region.x_min, y0, region.x_max, region.y_max)
                )
            else:
                y1 = y0 + heights[i]
                sub_regions.append(Region(region.x_min, y0, region.x_max, y1))
                y0 = y1 + gap
        return sub_regions

    def _stack_vertical_stripes(
        self,
        groups: List[Set[str]],
        group_areas: List[float],
        region: Region,
    ) -> List[Region]:
        """
        Full-height columns separated by ``macro_gap``; each column width is at least
        the widest packed footprint in that cluster, with extra width split by area weights.
        """
        n = len(groups)
        gap = self.macro_gap
        T = sum(group_areas)
        if T <= 0.0:
            T = 1e-9
        mins = [self._min_packed_width_in_group(g) for g in groups]
        gaps_total = (n - 1) * gap
        W_eff = max(0.0, region.width - gaps_total)
        slack = W_eff - sum(mins)
        if slack < -1e-6:
            debug_print(
                f"_stack_vertical_stripes: region width {region.width:.3f} um cannot "
                f"fit min column widths {sum(mins):.3f} um plus {gaps_total:.3f} um gutters; "
                "using area-only split (placement may be tight)."
            )
            widths = [W_eff * (group_areas[i] / T) for i in range(n)]
        else:
            widths = [
                mins[i] + max(0.0, slack) * (group_areas[i] / T) for i in range(n)
            ]
        x0 = region.x_min
        sub_regions: List[Region] = []
        for i in range(n):
            if i == n - 1:
                sub_regions.append(
                    Region(x0, region.y_min, region.x_max, region.y_max)
                )
            else:
                x1 = x0 + widths[i]
                sub_regions.append(Region(x0, region.y_min, x1, region.y_max))
                x0 = x1 + gap
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
    #  Fallback: guaranteed non-overlap packing
    # ──────────────────────────────────────────────────────────

    def _macro_bbox_um(
        self,
        node: str,
        ll: Tuple[float, float],
    ) -> Tuple[float, float, float, float]:
        """Axis-aligned LEF bbox (xmin, ymin, xmax, ymax) for node at lower-left ll."""
        x, y = ll
        w, h = self._node_size[node]
        return (x, y, x + w, y + h)

    def _pair_bbox_overlap(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
        clearance: float,
    ) -> bool:
        """True if expanded boxes overlap (clearance on each side, like minimum gap)."""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (
            ax1 + clearance <= bx0
            or bx1 + clearance <= ax0
            or ay1 + clearance <= by0
            or by1 + clearance <= ay0
        )

    def _layout_has_bbox_violations(
        self,
        positions: Dict[str, Tuple[float, float]],
        clearance: float,
    ) -> bool:
        """True if any two LEF bboxes violate ``clearance`` (overlap or closer than required gap)."""
        nodes = list(positions.keys())
        for i, na in enumerate(nodes):
            ba = self._macro_bbox_um(na, positions[na])
            for nb in nodes[i + 1 :]:
                bb = self._macro_bbox_um(nb, positions[nb])
                if self._pair_bbox_overlap(ba, bb, clearance):
                    return True
        return False

    def _pack_uniform_grid_no_overlap(
        self,
        nodes: List[str],
        region: Region,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Place macros on a uniform grid with cell size at least each macro's envelope
        when the region allows it. Avoids independent per-macro clamping that stacked
        many macros on one corner (overlap) in row packing.
        """
        n = len(nodes)
        if n == 0:
            return {}
        gap = self.macro_gap
        max_ew = max(self._effective_macro_envelope(node)[0] for node in nodes)
        max_eh = max(self._effective_macro_envelope(node)[1] for node in nodes)

        cols = n
        found = False
        for try_cols in range(1, n + 1):
            rows = max(1, math.ceil(n / try_cols))
            sw, sh = self._gapped_slot_dimensions(region, try_cols, rows)
            if sw + 1e-9 >= max_ew and sh + 1e-9 >= max_eh:
                cols = try_cols
                found = True
                break

        rows = max(1, math.ceil(n / cols))
        slots = self._grid_slots_for_shape(region, cols, rows)

        sorted_nodes = sorted(nodes, key=lambda nn: self._node_area.get(nn, 0), reverse=True)
        positions: Dict[str, Tuple[float, float]] = {}
        for idx, node in enumerate(sorted_nodes):
            if idx >= len(slots):
                break
            sx, sy, sw, sh = slots[idx]
            nw, nh = self._node_size[node]
            x = sx + max(0.0, (sw - nw) / 2.0)
            y = sy + max(0.0, (sh - nh) / 2.0)
            x, y = self._clamp_macro_bbox_ll(
                x, y, nw, nh,
                region.x_min, region.y_min, region.x_max, region.y_max,
            )
            positions[node] = (x, y)

        if not found:
            positions = self._legalize_push_apart(positions, region, clearance=gap)

        return positions

    def _min_grid_multiple_um(self, min_um: float) -> float:
        """At least ``min_um``, rounded up to a whole number of manufacturing-grid steps."""
        g = self.manufacturing_grid
        if g <= 0.0 or min_um <= 0.0:
            return max(min_um, 0.0)
        n = max(1, math.ceil(min_um / g))
        return n * g

    def _snap_macro_ll_in_region(
        self,
        node: str,
        x: float,
        y: float,
        region: Region,
    ) -> Tuple[float, float]:
        """Snap lower-left to manufacturing grid and clamp LEF bbox inside ``region``."""
        w, h = self._node_size[node]
        sx = self._snap(x)
        sy = self._snap(y)
        return self._clamp_macro_bbox_ll(
            sx,
            sy,
            w,
            h,
            region.x_min,
            region.y_min,
            region.x_max,
            region.y_max,
        )

    def _snap_all_macro_positions(
        self,
        positions: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Same snap + clamp sequence as ``place()`` uses after recursion, so legalization
        cannot leave origins off the manufacturing grid.
        """
        out: Dict[str, Tuple[float, float]] = {}
        fp = self._active_placement_region
        for node, (x, y) in positions.items():
            x, y = self._snap(x), self._snap(y)
            w, h = self._node_size[node]
            x, y = self._clamp_macro_bbox_ll(
                x, y, w, h, fp.x_min, fp.y_min, fp.x_max, fp.y_max,
            )
            if self._die_region is not None:
                m = self.die_inner_margin_um
                dr = self._die_region
                x, y = self._clamp_macro_bbox_ll(
                    x, y, w, h, dr.x_min + m, dr.y_min + m, dr.x_max - m, dr.y_max - m,
                )
            x, y = self._snap(x), self._snap(y)
            x, y = self._clamp_macro_bbox_ll(
                x, y, w, h, fp.x_min, fp.y_min, fp.x_max, fp.y_max,
            )
            if self._die_region is not None:
                m = self.die_inner_margin_um
                dr = self._die_region
                x, y = self._clamp_macro_bbox_ll(
                    x, y, w, h, dr.x_min + m, dr.y_min + m, dr.x_max - m, dr.y_max - m,
                )
            out[node] = (x, y)
        return out

    def _legalize_push_apart(
        self,
        positions: Dict[str, Tuple[float, float]],
        region: Region,
        clearance: float,
        max_iters: int = 64,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Resolve pairwise bbox overlaps by pushing macros apart while staying in region.
        Moves are whole multiples of the manufacturing grid so pin shapes stay on-grid.
        """
        pos = {n: self._snap_macro_ll_in_region(n, x, y, region) for n, (x, y) in positions.items()}
        # Heavier macros first as na so the inner nb tends to be smaller and is moved first.
        nodes = sorted(pos.keys(), key=lambda nn: self._node_area.get(nn, 0.0), reverse=True)
        eps = 1e-6
        step_um = self._min_grid_multiple_um(clearance)
        for _ in range(max_iters):
            moved = False
            for i, na in enumerate(nodes):
                for nb in nodes[i + 1 :]:
                    xa, ya = pos[na]
                    xb, yb = pos[nb]
                    ba = self._macro_bbox_um(na, (xa, ya))
                    bb = self._macro_bbox_um(nb, (xb, yb))
                    if not self._pair_bbox_overlap(ba, bb, clearance):
                        continue
                    wa, ha = self._node_size[na]
                    wb, hb = self._node_size[nb]
                    cx_a = xa + wa / 2.0
                    cy_a = ya + ha / 2.0
                    cx_b = xb + wb / 2.0
                    cy_b = yb + hb / 2.0
                    dx = cx_b - cx_a
                    dy = cy_b - cy_a
                    if abs(dx) + abs(dy) < eps:
                        dx = 1.0
                        dy = 0.0
                    pair_moved = False
                    if abs(dx) >= abs(dy):
                        delta = step_um if dx > 0 else -step_um
                        nxb = self._snap(xb + delta)
                        nxb, yb2 = self._snap_macro_ll_in_region(nb, nxb, yb, region)
                        if abs(nxb - xb) > eps or abs(yb2 - yb) > eps:
                            pos[nb] = (nxb, yb2)
                            pair_moved = True
                        else:
                            nxa = self._snap(xa - delta)
                            nxa, ya2 = self._snap_macro_ll_in_region(na, nxa, ya, region)
                            if abs(nxa - xa) > eps or abs(ya2 - ya) > eps:
                                pos[na] = (nxa, ya2)
                                pair_moved = True
                    else:
                        delta = step_um if dy > 0 else -step_um
                        nyb = self._snap(yb + delta)
                        xb2, nyb = self._snap_macro_ll_in_region(nb, xb, nyb, region)
                        if abs(nyb - yb) > eps or abs(xb2 - xb) > eps:
                            pos[nb] = (xb2, nyb)
                            pair_moved = True
                        else:
                            nya = self._snap(ya - delta)
                            xa2, nya = self._snap_macro_ll_in_region(na, xa, nya, region)
                            if abs(nya - ya) > eps or abs(xa2 - xa) > eps:
                                pos[na] = (xa2, nya)
                                pair_moved = True
                    if pair_moved:
                        moved = True
            if not moved:
                break
        return pos

    def _legalize_global_overlaps(
        self,
        positions: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Remove overlaps using the active placement window (centered subset or full floorplan)."""
        return self._legalize_push_apart(
            positions,
            self._active_placement_region,
            clearance=self.macro_gap,
            max_iters=256,
        )

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

    def _def_dbu_ll_from_micron(self, um: float) -> int:
        """
        Convert a micron coordinate to integer DEF DBU on the manufacturing grid.

        OpenROAD requires instance origins (and thus pin shapes built from LEF offsets)
        to land on the tech manufacturing grid in DBU; snapping only in float microns
        can leave ``round(um * dbu)`` one DBU off from a grid multiple.
        """
        d = self.dbu_per_micron
        if self.manufacturing_grid <= 0:
            return int(round(um * d))
        gdbu = max(1, int(round(self.manufacturing_grid * d)))
        dbu = int(round(um * d))
        return ((dbu + gdbu // 2) // gdbu) * gdbu

    def _snap(self, value: float) -> float:
        """Snap microns so the lower-left maps to an integer DBU multiple of the manufacturing grid."""
        d = self.dbu_per_micron
        if d <= 0:
            if self.manufacturing_grid <= 0:
                return value
            return round(value / self.manufacturing_grid) * self.manufacturing_grid
        return self._def_dbu_ll_from_micron(value) / d

    def _layout_expansion_half_um(self, node: str) -> float:
        """
        Uniform internal per-side padding used only for partitioning and slot feasibility.
        This preserves the previous logic behavior while removing physical halo requirements.
        """
        return self._internal_layout_padding_um

    def _node_effective_area(self, node: str) -> float:
        """Area used for hierarchical partitioning (LEF area plus internal logic padding)."""
        w, h = self._node_size.get(node, (0.0, 0.0))
        m = self._layout_expansion_half_um(node)
        return (w + 2.0 * m) * (h + 2.0 * m)
