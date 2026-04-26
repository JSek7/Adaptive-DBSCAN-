import torch
import math
from typing import Tuple    
import numpy as np 


class Adaptive_DBSCAN:
    def __init__(self, a: float = 0.03, b: float = 0.2, K: int = 1000, min_samples_min: int = 5, min_samples_max: int = 200, device=None):
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #coefficients (tunable on real data)
        self.a = a
        self.b = b
        self.min_samples_min = min_samples_min
        self.min_samples_max = min_samples_max
        self.K = K




    def params(self, points: torch.Tensor):

        distances = torch.norm(points, p=2, dim=1).to(device=self.device)
        min_samples = torch.ceil(self.K / (distances**2)).to(torch.int64).to(device=self.device)
        min_samples = torch.maximum(min_samples, torch.tensor(self.min_samples_min, device=self.device))
        min_samples = torch.minimum(min_samples, torch.tensor(self.min_samples_max, device=self.device))
       
        epsilons = self.a * distances + self.b

        return epsilons, min_samples

    def grid_coords(self, points, cell: float):
        # points: (N, 2) float tensor
        # returns cx, cy: (N,) long tensors (integer cell coordinates)
        cx = torch.floor(points[:, 0] / cell).long()
        cy = torch.floor(points[:, 1] / cell).long()
        return cx, cy


    def build_edges_grid_adaptive_eps_sorted(self,
        points: torch.Tensor,
        eps_i: torch.Tensor,
        cell_quantile: float = 0.90,
        min_cell: float = 0.2,
        block: int = 4096,
        cap_R_at_1: bool = True,   # set False if you really need larger R
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = points.device
        N = points.shape[0]

        eps_max = eps_i.max().item()

        # quantile on-device (avoid CPU roundtrip)
        cell = torch.quantile(eps_i.detach(), cell_quantile).item()
        cell = max(cell, min_cell)

        if cap_R_at_1:
            cell = max(cell, eps_max)  # ensures R=1

        R = max(1, int(math.ceil(eps_max / cell)))
        if cap_R_at_1:
            R = min(R, 1)

        # integer cell coords (on device)
        cx = torch.floor(points[:, 0] / cell).to(torch.int64)
        cy = torch.floor(points[:, 1] / cell).to(torch.int64)

        # pack (cx,cy) -> single int64 cell id
        # choose a stride big enough so ids don't collide
        # (you can also use hashing, but this is simple & fast)
        cy_min = cy.min()
        cy_shift = cy - cy_min
        stride = (cy_shift.max().item() + 1) + (2 * R + 3)  # safe margin
        cell_id = cx * stride + cy_shift

        # sort points by cell_id once
        order = torch.argsort(cell_id)
        cell_sorted = cell_id[order]

        # find unique cells and [start,end) ranges in the sorted array
        uniq, counts = torch.unique_consecutive(cell_sorted, return_counts=True)
        starts = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0)
        ends = starts + counts

        # move uniq to CPU for dict lookup (small: ~#cells)
        uniq_cpu = uniq.detach().cpu().tolist()
        starts_cpu = starts.detach().cpu().tolist()
        ends_cpu = ends.detach().cpu().tolist()

        ranges = {cid: (s, e) for cid, s, e in zip(uniq_cpu, starts_cpu, ends_cpu)}

        # precompute offset list
        offsets = [(dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1)]

        src_chunks = []
        dst_chunks = []

        # iterate over occupied cells (still Python loop, but far fewer expensive ops inside)
        for cid, s, e in zip(uniq_cpu, starts_cpu, ends_cpu):
            I = order[s:e]          # (a,) indices of points in this cell
            Pi = points[I]          # (a,2)
            eps2_i = (eps_i[I] ** 2)[:, None]  # (a,1)

            gx = cid // stride
            gy_shift = cid - gx * stride

            # gather neighbor indices by slicing the sorted order tensor
            neigh_slices = []
            for dx, dy in offsets:
                ncx = gx + dx
                ncy_shift = gy_shift + dy
                ncid = int(ncx * stride + ncy_shift)
                r = ranges.get(ncid)
                if r is not None:
                    ns, ne = r
                    neigh_slices.append(order[ns:ne])

            if not neigh_slices:
                continue

            J = torch.cat(neigh_slices, dim=0)   # (b,)
            Pj = points[J]                       # (b,2)

            # blocked compare over J to avoid OOM
            b = Pj.shape[0]
            for start in range(0, b, block):
                end = min(start + block, b)
                Pjb = Pj[start:end]     # (bb,2)
                Jb  = J[start:end]      # (bb,)

                # use cdist (often faster than manual broadcast)
                dist2 = torch.cdist(Pi, Pjb, p=2) ** 2   # (a,bb)
                mask = dist2 <= eps2_i                   # (a,bb)

                ii, jj = torch.where(mask)
                if ii.numel():
                    src_chunks.append(I[ii])
                    dst_chunks.append(Jb[jj])

        if not src_chunks:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        src = torch.cat(src_chunks, dim=0)
        dst = torch.cat(dst_chunks, dim=0)

        keep = src != dst
        return src[keep], dst[keep]


    def dbscan_from_edges(self, N, src, dst, min_samples, max_iters=50):
        device = src.device

        # 1) neighbour counts
        counts = torch.bincount(src, minlength=N)
        
        # 2) core points
        core = counts >= min_samples

        # 3) keep only core-core neighbour relations (for forming core clusters)
        mask_cc = core[src] & core[dst]
        cc_src0 = src[mask_cc]
        cc_dst0 = dst[mask_cc]

        cc_src = torch.cat([cc_src0, cc_dst0], dim=0)
        cc_dst = torch.cat([cc_dst0, cc_src0], dim=0)


        # 4) label spreading among core points
        labels = torch.full((N,), -1, device=device, dtype=torch.long)
        labels[core] = torch.arange(N, device=device, dtype=torch.long)[core]

        for _ in range(max_iters):
            old = labels
            cand = old[cc_src]
            new = old.clone()
            new.scatter_reduce_(0, cc_dst, cand, reduce="amin", include_self=True)
            labels = new
            if torch.equal(labels, old):
                break

        # 5) compress root labels into 0..K-1 cluster IDs (cores only)
        cluster = torch.full((N,), -1, device=device, dtype=torch.long)
        roots = labels[core]
        if roots.numel() > 0:
            unique_roots, inv = torch.unique(roots, sorted=True, return_inverse=True)
            cluster[core] = inv

        # 6) attach border points: core -> noncore
        mask_cb = (~core[src]) & (core[dst])
        cb_src = src[mask_cb]   # core points
        cb_dst = dst[mask_cb]   # border candidates

        if cb_dst.numel() > 0:
            cand_clusters = cluster[cb_src]      # cluster of the core
            cluster2 = cluster.clone()
            cluster2.scatter_reduce_(0, cb_dst, cand_clusters, reduce="amin", include_self=True)
            cluster = cluster2


        return cluster, core, counts


   