import numpy as np
import math
from typing import List


class Vector3D:
    """轻量三维向量。"""

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, s: float) -> 'Vector3D':
        return Vector3D(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> 'Vector3D':
        return self.__mul__(s)

    def __neg__(self) -> 'Vector3D':
        return Vector3D(-self.x, -self.y, -self.z)

    def __repr__(self) -> str:
        return f"Vector3D({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)


class BoundedPlane3D:
    """
    有界矩形面（棱柱的一个侧面）。

    顶点顺序（从外侧看逆时针）:
        vertices = [p0, p1, p2, p3]
        p0: 右下底, p1: 左下底, p2: 左上顶, p3: 右上顶

    外法向量 = normalize(u_vec × v_vec)
        u_vec = p1 - p0（向左）, v_vec = p3 - p0（向上）
        left × up = forward（指向外侧）
    """

    def __init__(self, vertices: List[Vector3D]):
        if len(vertices) != 4:
            raise ValueError("矩形面需要恰好 4 个顶点")
        self.vertices = vertices

        p0, p1, p2, p3 = vertices
        self.u_vec = p1 - p0
        self.v_vec = p3 - p0
        self.u_len_sq = self.u_vec.dot(self.u_vec)
        self.v_len_sq = self.v_vec.dot(self.v_vec)

        self.normal = self.u_vec.cross(self.v_vec).normalize()
        A, B, C = self.normal.x, self.normal.y, self.normal.z
        D = -self.normal.dot(p0)
        self.plane_eq = (A, B, C, D)

        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]
        zs = [v.z for v in vertices]
        self.bounding_box = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    def contains_point(self, p: Vector3D, tol: float = 1e-6) -> bool:
        """判断点 p 是否落在矩形面内。"""
        A, B, C, D = self.plane_eq

        if abs(A * p.x + B * p.y + C * p.z + D) > tol:
            return False

        xmin, xmax, ymin, ymax, zmin, zmax = self.bounding_box
        if not (xmin - tol <= p.x <= xmax + tol
                and ymin - tol <= p.y <= ymax + tol
                and zmin - tol <= p.z <= zmax + tol):
            return False

        rel = p - self.vertices[0]
        u = rel.dot(self.u_vec) / self.u_len_sq
        v = rel.dot(self.v_vec) / self.v_len_sq
        return -tol <= u <= 1.0 + tol and -tol <= v <= 1.0 + tol

    def generate_sample_points(self, density: int = 15) -> List[Vector3D]:
        """在矩形面内均匀采样，避开边缘。"""
        p0 = self.vertices[0]
        u_vals = np.linspace(0.02, 0.98, density)
        v_vals = np.linspace(0.02, 0.98, density)
        points = []
        for u in u_vals:
            for v in v_vals:
                points.append(p0 + u * self.u_vec + v * self.v_vec)
        return points


class TriangularPrism:
    """
    直立三棱柱（等边三角形截面）。

    面序号约定 — 严格匹配 AngleGenerator.generate() 的输出:
        Face 1 (sides[0]): 法向量方位角 = rotation
        Face 2 (sides[1]): 法向量方位角 = rotation + 120°
        Face 3 (sides[2]): 法向量方位角 = rotation + 240°

    几何构造:
        三个顶点方位角分别为 rotation-60°, rotation+60°, rotation+180°
        Face i 由顶点 i 和顶点 i+1 之间的边构成，
        使其外法向量恰好指向 rotation + i×120°。
    """

    def __init__(self, center_x: float, center_y: float,
                 width: float, side_height: float,
                 base_height: float = 0.0, rotation: float = 0.0):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.side_height = side_height
        self.base_height = base_height
        self.rotation = rotation
        self.sides = self._create_sides()

    def _create_sides(self) -> List[BoundedPlane3D]:
        R = self.width / math.sqrt(3.0)

        # 三个顶点方位角: rotation - 60°, rotation + 60°, rotation + 180°
        vertex_angles = [
            math.radians(self.rotation - 60.0 + k * 120.0) for k in range(3)
        ]

        vertices_2d = []
        for angle in vertex_angles:
            x = self.center_x + R * math.sin(angle)
            y = self.center_y + R * math.cos(angle)
            vertices_2d.append((x, y))

        sides = []
        for i in range(3):
            j = (i + 1) % 3
            # Face i 连接顶点 j（右侧）和顶点 i（左侧），从外侧看
            # p0 = 右下, p1 = 左下, p2 = 左上, p3 = 右上
            x_r, y_r = vertices_2d[j]  # 右侧顶点
            x_l, y_l = vertices_2d[i]  # 左侧顶点

            p0 = Vector3D(x_r, y_r, self.base_height)
            p1 = Vector3D(x_l, y_l, self.base_height)
            p2 = Vector3D(x_l, y_l, self.base_height + self.side_height)
            p3 = Vector3D(x_r, y_r, self.base_height + self.side_height)

            sides.append(BoundedPlane3D([p0, p1, p2, p3]))
        return sides

    def get_all_faces(self) -> List[BoundedPlane3D]:
        return self.sides


class CalShade:
    """阴影计算引擎。

    对每个面用双线性采样生成探测点，沿指向太阳方向做射线追踪，
    判断是否被其他棱柱遮挡。被遮挡的采样点比例即为阴影比。
    """

    def __init__(self):
        pass

    def compute_single_time_shadows(
        self,
        prisms_params: List[dict],
        solar_elevation: float,
        solar_azimuth: float,
        incidence_angles: List[float],
        sample_density: int = 15,
    ) -> List[dict]:

        prisms = [TriangularPrism(**param) for param in prisms_params]
        all_faces: List[BoundedPlane3D] = []
        face_metadata: List[dict] = []

        for p_idx, prism in enumerate(prisms):
            for f_idx, face in enumerate(prism.get_all_faces()):
                all_faces.append(face)
                face_metadata.append({
                    "prism_index": p_idx + 1,
                    "face_index": f_idx + 1,
                    "total_area": prism.width * prism.side_height,
                })

        if len(incidence_angles) != len(all_faces):
            raise ValueError(
                f"入射角数组长度({len(incidence_angles)})与总面数({len(all_faces)})不匹配"
            )

        # 夜间
        if solar_elevation <= 0:
            return [
                {
                    "Prism": m["prism_index"],
                    "Face": m["face_index"],
                    "Shadow Area (m2)": round(m["total_area"], 4),
                    "Shadow Ratio (%)": 100.0,
                    "Status": "Night",
                }
                for m in face_metadata
            ]

        alt_rad = math.radians(solar_elevation)
        az_rad = math.radians(solar_azimuth)

        light_dir = Vector3D(
            math.sin(az_rad) * math.cos(alt_rad),
            math.cos(az_rad) * math.cos(alt_rad),
            math.sin(alt_rad),
        ).normalize()

        results = []

        for idx, face in enumerate(all_faces):
            meta = face_metadata[idx]
            aoi = incidence_angles[idx]
            total_area = meta["total_area"]
            current_prism = meta["prism_index"]

            # AOI >= 90° 表示该面背对太阳，完全自遮挡
            if aoi >= 90.0:
                results.append({
                    "Prism": current_prism,
                    "Face": meta["face_index"],
                    "Shadow Area (m2)": round(total_area, 4),
                    "Shadow Ratio (%)": 100.0,
                    "Status": "Self-Shadow",
                })
                continue

            sample_pts = face.generate_sample_points(sample_density)
            if not sample_pts:
                shadow_ratio = 0.0
            else:
                shadow_count = 0
                for pt in sample_pts:
                    if self._is_occluded(pt, light_dir, all_faces,
                                         face_metadata, current_prism):
                        shadow_count += 1
                shadow_ratio = shadow_count / len(sample_pts)

            shadow_area = total_area * shadow_ratio
            results.append({
                "Prism": current_prism,
                "Face": meta["face_index"],
                "Shadow Area (m2)": round(shadow_area, 4),
                "Shadow Ratio (%)": round(shadow_ratio * 100, 2),
                "Status": "Neighbor-Obstructed" if shadow_ratio > 0
                          else "Fully Illuminated",
            })

        return results

    def _is_occluded(
        self,
        pt: Vector3D,
        light_dir: Vector3D,
        all_faces: List[BoundedPlane3D],
        face_metadata: List[dict],
        current_prism: int,
    ) -> bool:
        """从 pt 沿 light_dir 做射线，检查是否被其他棱柱的任何面遮挡。"""
        for o_idx, other_face in enumerate(all_faces):
            if face_metadata[o_idx]["prism_index"] == current_prism:
                continue

            A, B, C, D = other_face.plane_eq
            denom = A * light_dir.x + B * light_dir.y + C * light_dir.z

            if abs(denom) < 1e-8:
                continue

            t = -(A * pt.x + B * pt.y + C * pt.z + D) / denom

            if t > 1e-6:
                hit = pt + light_dir * t
                if other_face.contains_point(hit):
                    return True
        return False


# ======================================================================
#  内置测试
# ======================================================================
if __name__ == "__main__":

    my_prisms = [
        {"center_x": 0.0, "center_y": 0.0, "width": 2.0,
         "side_height": 3.0, "rotation": 0},
        {"center_x": 2.5, "center_y": 0.0, "width": 2.0,
         "side_height": 3.0, "rotation": 0},
    ]

    test_elev = 38.83
    test_azim = 98.15

    test_incidence_angles = [
        76.95,
        57.80,
        139.34,
        76.95,
        57.80,
        139.34,
    ]

    calshade = CalShade()
    shadow_outputs = calshade.compute_single_time_shadows(
        prisms_params=my_prisms,
        solar_elevation=test_elev,
        solar_azimuth=test_azim,
        incidence_angles=test_incidence_angles,
        sample_density=100,
    )

    print(f"\n--- Shadow Results (Elev: {test_elev}°, Azim: {test_azim}°) ---")
    print(f"{'Prism':<8}{'Face':<8}{'Shadow Area (m2)':<20}"
          f"{'Shadow Ratio (%)':<20}{'Status':<25}")
    print("-" * 80)
    for res in shadow_outputs:
        print(f"{res['Prism']:<8}{res['Face']:<8}"
              f"{res['Shadow Area (m2)']:<20}"
              f"{res['Shadow Ratio (%)']:<20}"
              f"{res['Status']:<25}")
