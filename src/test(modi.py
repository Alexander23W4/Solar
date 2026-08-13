from AngleModule.AngleHandler import AngleHandler
from AssemblyModule.ComponentsHandler import AngleGenerator
from Shade.ShadeHandler import CalShade
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def get_prism_shadow_report(
    test_time, lat, lon, timezone, prisms_data, base_angle=0
):
    """
    Calculate and output the shadow area and ratio report, including AOI per face.
    """
    handler = AngleHandler(latitude=lat, longitude=lon, timeZone=timezone)
    shading_engine = CalShade()

    sol_pos = handler.getAngle(test_time)
    elevation = sol_pos['apparent_elevation'].iloc[0]
    azimuth = sol_pos['azimuth'].iloc[0]

    all_incidence_angles = []
    prism_gen = AngleGenerator(n=3, base_angle=base_angle)
    face_angles = prism_gen.generate()

    for _ in prisms_data:
        for angle in face_angles:
            aoi = handler.AngleCombination(test_time, angle)
            all_incidence_angles.append(aoi)

    shadow_results = shading_engine.compute_single_time_shadows(
        prisms_params=prisms_data,
        solar_elevation=elevation,
        solar_azimuth=azimuth,
        incidence_angles=all_incidence_angles
    )

    for i, res in enumerate(shadow_results):
        res['AOI'] = all_incidence_angles[i]

    return shadow_results, elevation, azimuth


def compute_solar_shadows_2026(
    lat=-62.12, lon=-58.57, timezone='Etc/GMT+3',
    prisms_data=None, base_angle=0, year=2026,
    sample_density=15, verbose=True, max_days=None,
):
    """
    计算 Solar 项目在给定年份每一天每个整点的三棱柱各面阴影面积。

    参数:
        lat, lon, timezone: 地理位置
        prisms_data: 三棱柱参数列表；若为 None 则使用默认双棱柱配置
            (中心 (0,0) 与 (0,14.98)，宽 9.98，高 11.5，rotation=base_angle)
        base_angle: 第 1 面法向量方位角
        year: 计算年份
        sample_density: 采样密度（与 ShadeHandler 默认一致）
        verbose: 是否打印进度

    返回:
        pandas.DataFrame，列:
            Date, Hour, Solar_Elevation, Solar_Azimuth,
            Prism, Face, AOI, Solar_Area, Solar_Status
    """
    if prisms_data is None:
        prisms_data = [
            {'center_x': 0.0, 'center_y': 0.0,
             'width': 9.98, 'side_height': 11.5, 'rotation': base_angle},
            {'center_x': 0.0, 'center_y': 14.98,
             'width': 9.98, 'side_height': 11.5, 'rotation': base_angle},
        ]

    handler = AngleHandler(latitude=lat, longitude=lon, timeZone=timezone)
    shading_engine = CalShade()
    prism_gen = AngleGenerator(n=3, base_angle=base_angle)
    face_angles = prism_gen.generate()

    results = []
    date = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    total_days = (end - date).days
    day_count = 0

    while date < end:
        if max_days is not None and day_count >= max_days:
            break
        for hour in range(24):
            time_str = date.replace(hour=hour).strftime('%Y-%m-%d %H:%M')

            try:
                sol_pos = handler.getAngle(time_str)
                elevation = float(sol_pos['apparent_elevation'].iloc[0])
                azimuth = float(sol_pos['azimuth'].iloc[0])
            except Exception:
                continue

            # 直接由公式计算 AOI，避免重复调用 pvlib
            all_incidence_angles = []
            for _ in prisms_data:
                for angle in face_angles:
                    if elevation > 0:
                        elev_rad = np.radians(elevation)
                        az_diff_rad = np.radians(azimuth - angle)
                        cos_val = np.cos(elev_rad) * np.cos(az_diff_rad)
                        cos_val = np.clip(cos_val, -1.0, 1.0)
                        aoi = float(np.degrees(np.arccos(cos_val)))
                    else:
                        aoi = -1000.0
                    all_incidence_angles.append(aoi)

            shadow_results = shading_engine.compute_single_time_shadows(
                prisms_params=prisms_data,
                solar_elevation=elevation,
                solar_azimuth=azimuth,
                incidence_angles=all_incidence_angles,
                sample_density=sample_density,
            )

            for i, res in enumerate(shadow_results):
                results.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Hour': hour,
                    'Solar_Elevation': round(elevation, 4),
                    'Solar_Azimuth': round(azimuth, 4),
                    'Prism': res['Prism'],
                    'Face': res['Face'],
                    'AOI': round(all_incidence_angles[i], 4),
                    'Solar_Area': res['Shadow Area (m2)'],
                    'Solar_Status': res['Status'],
                })

        date += timedelta(days=1)
        day_count += 1
        if verbose and day_count % 30 == 0:
            print(f"  [Solar] {day_count}/{total_days} 天完成")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("Solar 项目 2026 全年阴影计算")
    print("=" * 60)

    df = compute_solar_shadows_2026()
    output_csv = r'F:\南极国创\我的\solar_results.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n已保存 {len(df)} 行至 {output_csv}")
    print("\n前 12 行预览:")
    print(df.head(12).to_string(index=False))

    # 保留原示例用法（单时刻）
    print("\n" + "=" * 60)
    print("单时刻示例（原 test.py 用法）")
    print("=" * 60)
    lat, lon = 62.12, 58.57
    time_str = '2026-01-26 16:30'
    baseAngle = 180
    my_prisms = [
        {'center_x': 0.0, 'center_y': 0.0, 'width': 1.73,
         'side_height': 2.0, 'rotation': baseAngle},
        {'center_x': 0.0, 'center_y': 2.5, 'width': 1.73,
         'side_height': 2.0, 'rotation': baseAngle}
    ]
    results, elev, az = get_prism_shadow_report(
        test_time=time_str, lat=lat, lon=lon,
        timezone='Asia/Shanghai', prisms_data=my_prisms, base_angle=baseAngle
    )
    print(f"Time: {time_str} | Elevation: {elev:.2f}° | Azimuth: {az:.2f}°")
    for res in results:
        print(res)
