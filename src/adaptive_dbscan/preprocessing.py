import torch 
import numpy as np

# Preprocessor class for point cloud data - input is raw numpy array, output is preprocessed torch tensor ready for ground removal

class Preprocessor:
    def __init__(self, 
            voxel=False, 
            voxel_size=0.5, 
            mode="first",    
            x_min: float = 0.0,
            x_max: float = 20.0,
            y_min: float = -5.0,
            y_max: float = 5.0,
            z_min: float = -2.0,
            z_max: float = 3.0):
        self.voxel = voxel
        self.voxel_size = voxel_size
        self.mode = mode
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mode not in {"first", "centroid"}:
            raise ValueError("mode must be 'first' or 'centroid'")
        
    def validate_points(self, points: np.ndarray) -> None:
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape (N, 3) or (N, >=3)")

    def voxel_downsampling(self, points):
        points = points.to(self.device)

        if points.numel() == 0:
            return points
            
        s = max(float(self.voxel_size),1e-6)
        q = torch.floor(points/s).to(torch.int32)

        key = (q[:, 0].to(torch.int64) * 73856093) ^ \
            (q[:, 1].to(torch.int64) * 19349663) ^ \
            (q[:, 2].to(torch.int64) * 83492791)

        uniq,inv = torch.unique(key,return_inverse=True)
        inv = inv.to(self.device)

        V = uniq.numel()
        D = points.shape[1]
        N = points.shape[0]


        if self.mode == "centroid": 
            sums = torch.zeros((V,D), device = self.device, dtype=points.dtype)
            counts = torch.zeros((V,1), device = self.device, dtype=points.dtype)
            sums.index_add_(0, inv, points)
            ones = torch.ones((points.shape[0],1), device=self.device,dtype=points.dtype)
            counts.index_add_(0,inv,ones)

            centroids = sums/counts
            return centroids 

        if self.mode == "first":
            idx = torch.arange(N,device=self.device,dtype=torch.int64)
            first = torch.full((V,),N, device=self.device, dtype=torch.int64)
            first = first.scatter_reduce(0,inv,idx,reduce="amin",include_self=True)
            return points[first]
        
    def filter_points_by_range(self, points):
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        return points[mask]


    def preprocess(self, points: np.ndarray) -> torch.Tensor:
        self.validate_points(points)
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)

        points = self.filter_points_by_range(points)

        if self.voxel:
            points = self.voxel_downsampling(points)

        return points

