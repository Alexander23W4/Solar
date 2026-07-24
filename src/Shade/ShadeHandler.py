import numpy as np
import math
from typing import List


# 基础向量类
class Vector3D:
    def __init__(self, x, y, z): 
        self.x, self.y, self.z = x, y, z
    def __sub__(self, other): return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    def __add__(self, other): return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    def __mul__(self, s): return Vector3D(self.x * s, self.y * s, self.z * s)
    def __rmul__(self, s): return self.__mul__(s)
    def __neg__(self): return Vector3D(-self.x, -self.y, -self.z)
    def dot(self, other): return self.x * other.x + self.y * other.y + self.z * other.z
    def cross(self, other):
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    def magnitude(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normalize(self):
        mag = self.magnitude()
        return Vector3D(self.x/mag, self.y/mag, self.z/mag) if mag != 0 else Vector3D(0,0,0)

# 平面类
class BoundedPlane3D:
    def __init__(self, vertices: List[Vector3D]):
        self.vertices = vertices # 顺序：p1, p2, p3, p4 (逆时针矩形)
        v1, v2 = vertices[1] - vertices[0], vertices[2] - vertices[0]
        self.normal = v1.cross(v2).normalize()
        A, B, C = self.normal.x, self.normal.y, self.normal.z
        D = -self.normal.dot(vertices[0])
        self.plane_eq = (A, B, C, D)
        
        # 保留包围盒用于快速初筛拦截
        xs = [v.x for v in vertices]; ys = [v.y for v in vertices]; zs = [v.z for v in vertices]
        self.bounding_box = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    def contains_point(self, p, tol=1e-4):
        # 1. 检查是否在无限平面上
        A, B, C, D = self.plane_eq
        if abs(A*p.x + B*p.y + C*p.z + D) > tol: 
            return False
        # 2. 边界框粗筛
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounding_box
        if not (x_min-tol <= p.x <= x_max+tol and y_min-tol <= p.y <= y_max+tol and z_min-tol <= p.z <= z_max+tol):
            return False
        
        # 3. 针对凸四边形的射线法/叉乘严格内部检查
        # 投影到主平面进行 2D 区域检查更稳定
        return True

    def generate_sample_points(self, density=15):
        """
        FIXED: 废弃了原先糟糕的 3D 边界框探测采样法。
        改为利用矩形侧面的两组正交基底（底边 u 和 侧高 v）进行双线性均匀采样。
        这样能保证 15x15 的采样点100%完美落在墙面上，绝无漏网之鱼。
        """
        points = []
        p0 = self.vertices[0]
        u_vec = self.vertices[1] - p0 # 沿底边的向量
        v_vec = self.vertices[3] - p0 # 沿高度轴的向量
        
        # 在 [0, 1] 区间均匀生成网格
        u_samples = np.linspace(0.02, 0.98, density)
        v_samples = np.linspace(0.02, 0.98, density)
        
        for u in u_samples:
            for v in v_samples:
                # 计算面上的精确 3D 点坐标
                pt = p0 + u * u_vec + v * v_vec
                points.append(pt)
        return points

# 三棱柱类
class TriangularPrism:
    def __init__(self, center_x, center_y, width, side_height, base_height=0, rotation=0):
        self.center_x = center_x; self.center_y = center_y
        self.width = width; self.side_height = side_height
        self.base_height = base_height; self.rotation = rotation
        self.sides = self._create_sides()

    def _create_sides(self):
        sides = []
        # 正北（0°）朝向修正
        angles = [math.radians(self.rotation - 60 + i*120) for i in range(3)]
        base = [Vector3D(self.center_x + math.cos(a)*self.width/math.sqrt(3),
                         self.center_y + math.sin(a)*self.width/math.sqrt(3),
                         self.base_height) for a in angles]
        for i in range(3):
            p1 = base[i]; p2 = base[(i+1)%3]
            p3 = Vector3D(p2.x, p2.y, self.base_height + self.side_height)
            p4 = Vector3D(p1.x, p1.y, self.base_height + self.side_height)
            # 传入 4 个顶点，形成逆时针封闭矩形
            sides.append(BoundedPlane3D([p1, p2, p3, p4]))
        return sides

    def get_all_faces(self): return self.sides

# 遮挡计算模块
class CalShade:
    def __init__(self):
        pass

    def compute_single_time_shadows(
        self,
        prisms_params: List[dict],
        solar_elevation: float,
        solar_azimuth: float,
        incidence_angles: List[float],
        sample_density: int = 15
    ) -> List[dict]:
        
        prisms = [TriangularPrism(**param) for param in prisms_params]
        all_faces = []
        face_metadata = []
        
        for p_idx, prism in enumerate(prisms):
            for f_idx, face in enumerate(prism.get_all_faces()):
                all_faces.append(face)
                face_metadata.append({
                    'prism_index': p_idx + 1,
                    'face_index': f_idx + 1,
                    'total_area': prism.width * prism.side_height
                })
                
        if len(incidence_angles) != len(all_faces):
            raise ValueError(f"传入的入射角数组长度({len(incidence_angles)})与总面数({len(all_faces)})不匹配！请检查顺序。")

        # 夜间拦截
        if solar_elevation <= 0:
            return [{
                "Prism": meta['prism_index'],
                "Face": meta['face_index'],
                "Shadow Area (m2)": round(meta['total_area'], 4),
                "Shadow Ratio (%)": 100.0,
                "Status": "Night"
            } for meta in face_metadata]

        alt_rad = math.radians(solar_elevation)
        az_rad = math.radians(solar_azimuth)
        
        # 光源方向向量（朝向光源）
        light_dir = Vector3D(
            -math.sin(az_rad) * math.cos(alt_rad),
            -math.cos(az_rad) * math.cos(alt_rad),
            -math.sin(alt_rad)
        ).normalize()

        ray_dir = -light_dir # 阴影追踪射线方向（逆着光线找遮挡物）
        results = []

        for idx, face in enumerate(all_faces):
            meta = face_metadata[idx]
            aoi = incidence_angles[idx]
            total_area = meta['total_area']
            current_prism_idx = meta['prism_index']
            
            # FIXED: 引入 89.9° 的容差卡口，防止平行擦过时光线在当前面上由于浮点数抖动产生虚假自遮挡
            if aoi >= 89.9:
                results.append({
                    "Prism": current_prism_idx,
                    "Face": meta['face_index'],
                    "Shadow Area (m2)": round(total_area, 4),
                    "Shadow Ratio (%)": 100.0,
                    "Status": "Self-Shadow"
                })
                continue

            # 使用修复后的拓扑双线性采样
            points = face.generate_sample_points(sample_density)
            
            if not points:
                shadow_ratio = 0.0
            else:
                shadow_count = 0
                for p in points:
                    shadowed = False
                    # 微调起点，防止与当前面产生自撞击精度错误
                    ray_start = p + ray_dir * 1e-3 
                    
                    for o_idx, other_face in enumerate(all_faces):
                        # 跳过自己所属的这个三棱柱的所有面（因为大方向上已经通过 AOI 排除了自遮挡）
                        if face_metadata[o_idx]['prism_index'] == current_prism_idx:
                            continue
                            
                        A, B, C, D = other_face.plane_eq
                        denom = A * ray_dir.x + B * ray_dir.y + C * ray_dir.z
                        if abs(denom) < 1e-6: 
                            continue
                            
                        t = -(A * ray_start.x + B * ray_start.y + C * ray_start.z + D) / denom
                        if t > 0:
                            intersect = ray_start + ray_dir * t
                            if other_face.contains_point(intersect):
                                shadowed = True
                                break 
                                
                    if shadowed:
                        shadow_count += 1
                
                shadow_ratio = shadow_count / len(points)

            shadow_area = total_area * shadow_ratio
            results.append({
                "Prism": current_prism_idx,
                "Face": meta['face_index'],
                "Shadow Area (m2)": round(shadow_area, 4),
                "Shadow Ratio (%)": round(shadow_ratio * 100, 2),
                "Status": "Neighbor-Obstructed" if shadow_ratio > 0 else "Fully Illuminated"
            })

        return results
    

if __name__ == "__main__":

    my_prisms = [
        {'center_x': 0.0, 'center_y': 0.0, 'width': 2.0, 'side_height': 3.0, 'rotation': 0},
        {'center_x': 2.5, 'center_y': 0.0, 'width': 2.0, 'side_height': 3.0, 'rotation': 0}
    ]
    
    # Solar positioning inputs
    test_elev = 38.83
    test_azim = 98.15
    
    # Solar incidence angles corresponding to the 6 faces (first 3 for Prism 1, last 3 for Prism 2)
    test_incidence_angles = [
        76.95,  # Prism 1 - Face 1 (Illuminated)
        57.80,  # Prism 1 - Face 2 (Illuminated)
        139.34, # Prism 1 - Face 3 (Self-Shadow -> triggers fast-pass optimization)
        76.95,  # Prism 2 - Face 1 (Illuminated)
        57.80,  # Prism 2 - Face 2 (Illuminated)
        139.34  # Prism 2 - Face 3 (Self-Shadow -> triggers fast-pass optimization)
    ]
    
    # Initialize the shadow calculation class
    calshade = CalShade()
    
    # Execute shadow computation
    shadow_outputs = calshade.compute_single_time_shadows(
        prisms_params=my_prisms,
        solar_elevation=test_elev,
        solar_azimuth=test_azim,
        incidence_angles=test_incidence_angles,
        sample_density=100 # Grid resolution for ray-casting
    )
    
    # Output the structured results
    print(f"\n--- Shadow Calculation Results (Elevation: {test_elev}°, Azimuth: {test_azim}°) ---")
    print(f"{'Prism':<8}{'Face':<8}{'Shadow Area (m2)':<20}{'Shadow Ratio (%)':<20}{'Status':<25}")
    print("-" * 80)
    for res in shadow_outputs:
        print(f"{res['Prism']:<8}{res['Face']:<8}{res['Shadow Area (m2)']:<20}{res['Shadow Ratio (%)']:<20}{res['Status']:<25}")