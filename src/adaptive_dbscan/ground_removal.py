import numpy as np
import torch
import math


def fit_plane(
        p1,
        p2,
        p3,
        eps:float = 1e-12,
        require_vertical:  bool = True, 
        vertical_co_thresh: float = 0.85
): 
   
   
    v1 = p2 - p3
    v2 = p3 - p1

    n = torch.cross(v1,v2,dim=0)
    mag_n = torch.linalg.norm(n)
    n = n/(mag_n + eps) 

    if require_vertical:
        ok = (mag_n > eps) and (torch.abs(n[2]) >= vertical_co_thresh)
    else:
        ok = mag_n > eps

    d = -torch.dot(n,p1)

    return n,d,ok

def num_inliers(points, n, d,distance_thresh): 
    dist = (points@n + d).abs()
    mask_c = dist < distance_thresh
    inliers_c = int(mask_c.sum().item())
    return inliers_c,mask_c,dist

def refine_plane_svd(points: torch.Tensor, mask: torch.Tensor):
    """
    Refit plane using all inlier points via SVD.
    """
    inliers = points[mask]
    if inliers.shape[0] < 3:
        return None, None, False

    centroid = inliers.mean(dim=0)
    centered = inliers - centroid

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / (torch.linalg.norm(n) + 1e-12)

    d = -torch.dot(n, centroid)
    return n, d, True

def remove_ground(
    points: torch.Tensor,
    num_iters: int = 30,
    distance_thresh: float = 0.03,                
    lowest_z_frac: float = 0.5,
    seed: int = 0,) -> tuple[torch.Tensor, float, torch.Tensor]:
  
    
    #Check points dimensions and move to GPU
    assert points.ndim == 2 and points.shape[1] == 3 
    device = points.device 

    N = points.shape[0]

    #determine candidate idx for RANSAC 
    if lowest_z_frac is not None and 0 < lowest_z_frac < 1.0:
        z = points[:,2]
        K = max(3,int(math.ceil(N*lowest_z_frac)))

        cand_idx = torch.topk(z, k=K, largest=False).indices
    else: 
        cand_idx = torch.arange(N, device=device)
    
    cand_points = points[cand_idx]
    K = cand_points.shape[0]

    #random generator on GPU 
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    #best guess variables 
    best_inliers = 0
    best_n,best_d = None,None
    best_mask = None

    #RANSAC algo
    for i in range(num_iters): 
        s = torch.randperm(K, generator=g, device=device)[:3]
        p1,p2,p3 = cand_points[s[0]],cand_points[s[1]],cand_points[s[2]]

        n,d,ok = fit_plane(p1,p2,p3)

        if not bool(ok):
            continue 

        inliers_c,mask_c,__=num_inliers(points,n,d,distance_thresh) 
        if inliers_c > best_inliers:
            best_inliers = inliers_c
            best_n,best_d = n,d 

            best_mask = mask_c

    if best_mask is None:
        return points, None, None, None
    else:
        refined_n, refined_d, ok = refine_plane_svd(points, best_mask)
        if ok:
            best_n, best_d = refined_n, refined_d
            _, best_mask, _ = num_inliers(points, best_n, best_d, distance_thresh)

    non_ground = points[~best_mask]

    #remove z dimension

    non_ground = non_ground[:,:2]

    return non_ground,  best_n, best_d, best_mask

