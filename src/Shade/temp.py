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