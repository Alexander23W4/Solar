import numpy as np
import math
from typing import List
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pvlib

# -----------------------------三棱柱阴影计算相关类定义-------------------------------
# 基础向量类
class Vector3D:
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
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
        self.vertices = vertices
        v1, v2 = vertices[1] - vertices[0], vertices[2] - vertices[0]
        self.normal = v1.cross(v2).normalize()
        A,B,C = self.normal.x, self.normal.y, self.normal.z
        D = -self.normal.dot(vertices[0])
        self.plane_eq = (A,B,C,D)
        xs = [v.x for v in vertices]; ys = [v.y for v in vertices]; zs = [v.z for v in vertices]
        self.bounding_box = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    def contains_point(self, p, tol=1e-6):
        A,B,C,D = self.plane_eq
        if abs(A*p.x + B*p.y + C*p.z + D) > tol: return False
        x_min,x_max,y_min,y_max,z_min,z_max = self.bounding_box
        if not (x_min-tol <= p.x <= x_max+tol and y_min-tol <= p.y <= y_max+tol and z_min-tol <= p.z <= z_max+tol):
            return False
        return True

    def generate_sample_points(self, density=5):
        x_min,x_max,y_min,y_max,z_min,z_max = self.bounding_box
        points = []
        xs = np.linspace(x_min,x_max,density)
        ys = np.linspace(y_min,y_max,density)
        zs = np.linspace(z_min,z_max,density)
        for x in xs:
            for y in ys:
                for z in zs:
                    p = Vector3D(x,y,z)
                    if self.contains_point(p): points.append(p)
        return points

# ---- 三棱柱类 ----
class TriangularPrism:
    def __init__(self, center_x, center_y, width, side_height, base_height=0, rotation=0):
        self.center_x = center_x; self.center_y = center_y
        self.width = width; self.side_height = side_height
        self.base_height = base_height; self.rotation = rotation
        self.sides = self._create_sides()

    def _create_sides(self):
        sides = []
        angles = [math.radians(self.rotation + 90 + i*120) for i in range(3)]
        base = [Vector3D(self.center_x + math.cos(a)*self.width/math.sqrt(3),
                         self.center_y + math.sin(a)*self.width/math.sqrt(3),
                         self.base_height) for a in angles]
        for i in range(3):
            p1 = base[i]; p2 = base[(i+1)%3]
            p3 = Vector3D(p2.x,p2.y,self.base_height+self.side_height)
            p4 = Vector3D(p1.x,p1.y,self.base_height+self.side_height)
            sides.append(BoundedPlane3D([p1,p2,p3,p4]))
        return sides

    def get_all_faces(self): return self.sides


def compute_annual_energy_with_shadow(
    prisms_params: List[dict],
    latitude: float,
    longitude: float,
    year: int,
    timeZone: str = "UTC",
    step_minutes: int = 10,
    albedo: float = 0.85,
    ramp_interval_hours: int = 6,
    sample_density: int = 5
):
    # 创建三棱柱
    prisms = [TriangularPrism(**param) for param in prisms_params]
    all_faces = [face for prism in prisms for face in prism.get_all_faces()]

    # 生成全年时间序列
    times = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:59',
                          freq=f'{step_minutes}min', tz=timeZone)

    # 太阳位置 & 辐照度
    loc = pvlib.location.Location(latitude, longitude, tz=timeZone)
    solpos = loc.get_solarposition(times)
    apparent_elevations = solpos['apparent_elevation'].values
    azimuths = solpos['azimuth'].values
    clearsky = loc.get_clearsky(times, model='ineichen', linke_turbidity=3)
    DNIs = clearsky['dni'].values
    GHIs = clearsky['ghi'].values
    DHIs = clearsky['dhi'].values

    irradiances = []
    total_energy = 0.0

    # 计算单面辐照度
    def calc_face_energy(idx, face):
        apparent_elevation = apparent_elevations[idx]
        azimuth = azimuths[idx]
        DNI = DNIs[idx]; GHI = GHIs[idx]; DHI = DHIs[idx]
        if apparent_elevation <= 0 or (DNI <= 0 and GHI <= 0): return 0.0

        # 光照方向向量
        alt_rad = math.radians(apparent_elevation)
        az_rad = math.radians(azimuth)
        light_dir = Vector3D(-math.sin(az_rad)*math.cos(alt_rad),
                             -math.cos(az_rad)*math.cos(alt_rad),
                             -math.sin(alt_rad)).normalize()

        # 阴影计算
        points = face.generate_sample_points(sample_density)
        if not points: shadow_ratio = 1.0
        else:
            shadow_count = 0
            for p in points:
                shadowed = False
                ray_start = p - light_dir*10000
                for other_face in all_faces:
                    if other_face == face: continue
                    A,B,C,D = other_face.plane_eq
                    denom = A*light_dir.x + B*light_dir.y + C*light_dir.z
                    if abs(denom) < 1e-8: continue
                    t = -(A*ray_start.x + B*ray_start.y + C*ray_start.z + D)/denom
                    if 0 <= t <= 10000:
                        intersect = ray_start + light_dir*t
                        if other_face.contains_point(intersect):
                            shadowed = True
                            break
                if shadowed: shadow_count += 1
            shadow_ratio = 1.0 - shadow_count/len(points)

        # AOI 简化
        cos_theta = max(math.cos(alt_rad), 0.0)
        direct = DNI*cos_theta
        diffuse = DHI + albedo*GHI
        return float(shadow_ratio*(direct+diffuse))

    # 遍历时间点，多线程计算每个面
    for idx in range(len(times)):
        with ThreadPoolExecutor() as executor:
            panel_energies = list(executor.map(lambda f: calc_face_energy(idx,f), all_faces))
        minute_energy = float(sum(panel_energies)/len(all_faces))
        total_energy += minute_energy*step_minutes*60  # J/m²
        irradiances.append(minute_energy)

    irradiances = np.array(irradiances)
    irradiances = irradiances[irradiances>0]

    # 计算 CV
    mean_val = np.mean(irradiances)
    std_val = np.std(irradiances)
    cv = std_val/mean_val if mean_val>0 else 0

    # 计算 Ramp Rate
    step_per_ramp = int(ramp_interval_hours*60/step_minutes)
    ramp_averages = [np.mean(irradiances[i:i+step_per_ramp])
                     for i in range(0,len(irradiances),step_per_ramp) if len(irradiances[i:i+step_per_ramp])>0]
    ramp_averages = np.array(ramp_averages)
    ramp_rate = np.std(np.diff(ramp_averages)) if len(ramp_averages)>1 else 0.0

    return total_energy/3600000.0, cv, ramp_rate


