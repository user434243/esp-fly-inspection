#!/usr/bin/env python3
"""
planner.py - collision-free 3D paths through the bridge for the autonomous mission.

How it works
  1. The world (same boxes as the simulator) is turned into a 3D grid of 10 cm cells.
     A cell is 'blocked' if it is closer than CLEAR to any surface (drone size + margin).
  2. A* search finds the shortest path between two points through free cells.
     Cells close to walls cost extra, so paths stay in the middle of gaps.
  3. The path is shortened ("string pulling"): corners are cut wherever the straight line
     stays in free space, giving a few straight legs instead of hundreds of grid steps.
"""
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

RES = 0.10                                   # grid cell size (m)
CLEAR = 0.14                                 # drone centre must stay this far from any surface
LO = np.array([-1.5, -11.5, 0.0])            # grid corner (m)
HI = np.array([9.5, 17.0, 4.4])
Z_MIN, Z_MAX = 0.30, 4.20                    # never below / above (ground effect, ceiling)
WALL_COST, WALL_NEAR = 3.0, 0.35             # extra cost when closer than WALL_NEAR to a wall


class Planner:
    def __init__(self, boxes, lo=None, hi=None, clear=None):   # --- big worlds (patch_world2) ---
        self.lo = np.array(LO if lo is None else lo, float)        # grid corners: the world's size
        self.hi = np.array(HI if hi is None else hi, float)
        z_max = Z_MAX if hi is None else self.hi[2] - 0.2
        CLEAR = globals()["CLEAR"] if clear is None else clear   # --- wind margin (patch_margin) ---
        self.clear = CLEAR
        self.shape = tuple(int(round(v)) for v in (self.hi - self.lo) / RES)
        nx, ny, nz = self.shape
        cx = self.lo[0] + (np.arange(nx) + 0.5) * RES
        cy = self.lo[1] + (np.arange(ny) + 0.5) * RES
        cz = self.lo[2] + (np.arange(nz) + 0.5) * RES
        self.axes = (cx, cy, cz)
        blocked = np.zeros(self.shape, bool)
        for lo, hi in boxes:                   # distance from the cells NEAR the box to the box
            i0, i1 = np.searchsorted(cx, lo[0] - CLEAR), np.searchsorted(cx, hi[0] + CLEAR)
            j0, j1 = np.searchsorted(cy, lo[1] - CLEAR), np.searchsorted(cy, hi[1] + CLEAR)
            k0, k1 = np.searchsorted(cz, lo[2] - CLEAR), np.searchsorted(cz, hi[2] + CLEAR)
            if i0 >= i1 or j0 >= j1 or k0 >= k1:
                continue
            sx, sy, sz = cx[i0:i1], cy[j0:j1], cz[k0:k1]
            dx = np.maximum(np.maximum(lo[0] - sx, sx - hi[0]), 0)[:, None, None]
            dy = np.maximum(np.maximum(lo[1] - sy, sy - hi[1]), 0)[None, :, None]
            dz = np.maximum(np.maximum(lo[2] - sz, sz - hi[2]), 0)[None, None, :]
            blocked[i0:i1, j0:j1, k0:k1] |= (dx ** 2 + dy ** 2 + dz ** 2) < CLEAR ** 2
        blocked |= (cz < Z_MIN)[None, None, :] | (cz > z_max)[None, None, :]
        blocked[0, :, :] = blocked[-1, :, :] = True        # border: keeps neighbour indices valid
        blocked[:, 0, :] = blocked[:, -1, :] = True
        blocked[:, :, 0] = blocked[:, :, -1] = True
        self.blocked = blocked
        clearance = distance_transform_edt(~blocked) * RES   # metres to the nearest blocked cell
        self.clearance = clearance
        pen = 1.0 + WALL_COST * np.clip((WALL_NEAR - clearance) / WALL_NEAR, 0, 1)
        self._pen = pen.ravel()
        self._blk = blocked.ravel()
        sx, sy = ny * nz, nz
        self._nb = [(dx * sx + dy * sy + dz, math.sqrt(dx * dx + dy * dy + dz * dz))
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dy or dz]

    # ---- helpers ----
    def cell(self, p):
        return tuple(int(np.clip((p[k] - self.lo[k]) / RES, 1, self.shape[k] - 2)) for k in range(3))

    def centre(self, c):
        return np.array([self.axes[k][c[k]] for k in range(3)])

    def free(self, p):
        return not self.blocked[self.cell(p)]

    def clear_at(self, p):
        return float(self.clearance[self.cell(p)])

    def line_free(self, a, b, step=0.04):
        n = max(1, int(np.linalg.norm(b - a) / step))
        return all(self.free(a + (b - a) * t) for t in np.linspace(0, 1, n + 1))

    def nearest_free(self, p):
        c = self.cell(p)
        if not self.blocked[c]:
            return c
        best = None
        for r in range(1, 6):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        q = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if all(1 <= q[k] < self.shape[k] - 1 for k in range(3)) and not self.blocked[q]:
                            d = dx * dx + dy * dy + dz * dz
                            if best is None or d < best[0]:
                                best = (d, q)
            if best:
                return best[1]
        raise RuntimeError(f"no free space near {p}")

    # ---- A* ----
    def astar(self, start, goal):
        nx, ny, nz = self.shape
        s = np.ravel_multi_index(self.nearest_free(start), self.shape)
        g_idx = np.ravel_multi_index(self.nearest_free(goal), self.shape)
        gx, gy, gz = np.unravel_index(g_idx, self.shape)
        sx = ny * nz
        best = {s: 0.0}
        parent = {s: -1}
        done = set()
        pen, blk, nb = self._pen, self._blk, self._nb
        heap = [(0.0, s)]
        while heap:
            _, u = heapq.heappop(heap)
            if u in done:
                continue
            if u == g_idx:
                break
            done.add(u)
            gu = best[u]
            for off, step in nb:
                v = u + off
                if blk[v] or v in done:
                    continue
                g = gu + step * pen[v]
                if g < best.get(v, 1e18):
                    best[v], parent[v] = g, u
                    x, r = divmod(v, sx)
                    y, z = divmod(r, nz)
                    h = math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (z - gz) ** 2)
                    heapq.heappush(heap, (g + 1.2 * h, v))
        if g_idx not in parent:
            raise RuntimeError(f"no path from {start} to {goal}")
        cells, u = [], g_idx
        while u != -1:
            cells.append(self.centre(np.unravel_index(u, self.shape)))
            u = parent[u]
        return cells[::-1]

    def smooth(self, pts):
        """String pulling: keep only the corners that are needed."""
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def plan(self, start, goal):
        """Straight legs from start to goal (both exact), avoiding the bridge."""
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if self.line_free(start, goal):
            return [start, goal]
        pts = self.astar(start, goal)
        pts = [start] + pts[1:-1] + [goal]
        return [np.asarray(p, float) for p in self.smooth(pts)]
