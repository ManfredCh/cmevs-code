"""
ERPT Core 模块

包含：
- tangent_extraction: ERP -> Tangent 切片生成
- depth_estimation: Depth Pro 深度估计
- depth_fusion: Tangent Depth -> ERP Depth 融合
- coordinate: 坐标系定义（锁定）
- erp_projection: ERP 投影（锁定）
"""
from .tangent_extraction import (
    TangentSlice,
    build_icosahedron_slices,
    extract_all_tangents,
    extract_tangent_from_erp,
    compute_coverage_mask,
    compute_ray_directions_for_slice,
)

from .depth_estimation import (
    DepthEstimator,
    estimate_all_tangent_depths,
)

from .depth_fusion import (
    fuse_tangent_depths_to_erp,
    visualize_depth,
    save_depth_visualization,
)

from .erp_warp import (
    WarpResult,
    warp_erp_to_target,
    create_comparison_image,
)
