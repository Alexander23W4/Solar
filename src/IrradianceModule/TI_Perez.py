import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
from datetime import datetime

from AngleModule.AngleHandler import AngleHandler
from IrradianceModule.IrradianceHandler import SolarIrradiance
from AssemblyModule.ComponentsHandler import AngleGenerator

"""
Perez 倾斜面辐照度转换模型 (Transposition Model) 物理逻辑说明
核心思想：打破传统模型将天空散射视为均匀分布的假设，将天空散射辐射细分为
三个具有不同空间分布特征的物理分量，并结合太阳直射和地表反射，计算倾斜
光伏阵列面 (POA) 上的总辐照度。

直射分量 (Beam Component)
基于几何光学（余弦效应）：
G_beam = DNI * cos(AOI)
其中 AOI 为太阳入射角 (Angle of Incidence)。

天空散射分量 (Sky Diffuse Component) [模型核心]
将水平散射辐射 (DHI) 在空间上解耦为三个独立的子分量，赋予不同几何权重：
G_sky = DHI * (F_iso + F_circumsolar + F_horizon)

各向同性分量 (F_iso)：
代表天空中均匀分布的散射光。接收比例取决于倾斜面"看到"的天空立体角，
几何权重为 (1 + cos(beta)) / 2，其中 beta 为倾斜角。

环日分量 (F_circumsolar)：
代表集中在太阳周围（约 3° 范围内）的强前向散射光。几何上视为与直射光
类似，按照太阳入射角 (AOI) 的余弦进行投影。

地平线亮带分量 (F_horizon)：
代表靠近地平线附近的亮带散射光（区别于 Hay 模型的关键创新）。根据太阳
天顶角及大气条件动态调整强度，并乘以倾斜面对地平线区域的视角系数。

动态参数化机制：
引入天空清晰度 (Epsilon, ε) 和天空亮度 (Delta, Δ) 两个无量纲参数，结合
太阳天顶角将天空划分为多种状态（全阴至极晴）。针对每种状态，利用全球气象站
实测数据预先标定的经验系数，动态计算 F_iso、F_circumsolar 和 F_horizon 的比例。

地面反射分量 (Ground Diffuse / Albedo Component)
假设地面为各向同性的朗伯反射 (Lambertian reflection)：
G_ground = GHI * albedo * (1 - cos(beta)) / 2
接收比例取决于倾斜面"看到"的地面立体角。

总倾斜面辐照度 (Total POA Irradiance)
将上述三个独立计算的分量进行线性叠加：
G_POA = G_beam + G_sky + G_ground
"""

def calculate_prism_power_perez(latitude, longitude, timezone, start_time, end_time, 
                                 n_faces=3, base_angle=0, face_area=1.0,
                                 surface_tilt=90, albedo=0.85,
                                 linke_turbidity=2.0, interval_hours=1):
    """
    使用 Perez 模型计算棱柱形光伏组件在指定时间范围内的总辐照度
    
    参数:
        latitude: 纬度 (度)
        longitude: 经度 (度)
        timezone: 时区, 如 'Asia/Shanghai'
        start_time: 开始时间字符串, 格式 'YYYY-MM-DD HH:MM'
        end_time: 结束时间字符串, 格式 'YYYY-MM-DD HH:MM'
        n_faces: 棱柱面数, 默认3
        base_angle: 第一个面的朝向角度 (度), 以正北为0°, 顺时针增加
        face_area: 每个面的面积 (m²), 默认1.0
        surface_tilt: 光伏板倾角 (度), 默认90° (垂直)
        albedo: 地面反照率, 默认0.85
        linke_turbidity: 林克浑浊度, 默认2.0
        interval_hours: 时间间隔 (小时), 默认1
    
    返回:
        dict: 包含总能量和各面的累计结果
    """
    # 初始化 AngleHandler 和 SolarIrradiance
    angle_handler = AngleHandler(
        latitude=latitude,
        longitude=longitude,
        timeZone=timezone
    )
    
    solar_irradiance = SolarIrradiance(
        latitude=latitude,
        longitude=longitude,
        tz=timezone,
        linke_turbidity=linke_turbidity
    )
    
    # 生成各面朝向
    angle_generator = AngleGenerator(n=n_faces, base_angle=base_angle)
    face_azimuths = angle_generator.generate()
    n_faces = len(face_azimuths)
    
    # 生成时间序列
    times = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f'{interval_hours}h',
        tz=timezone
    )
    
    if len(times) == 0:
        print("错误: 时间范围无效")
        return None
    
    print(f"=== Perez 模型计算 ===")
    print(f"纬度: {latitude:.3f}°, 经度: {longitude:.3f}°")
    print(f"时间范围: {start_time} 到 {end_time}")
    print(f"面数: {n_faces}, 倾角: {surface_tilt}°, 反照率: {albedo}")
    print(f"时间间隔: {interval_hours} 小时, 共 {len(times)} 个采样点\n")
    
    # 存储每个时刻的结果
    all_results = []
    face_energy = np.zeros(n_faces)
    face_valid_points = np.zeros(n_faces)
    total_energy = 0.0
    sun_above_count = 0
    
    # 进度记录
    progress_energy = []
    total_points = len(times)
    cumulative_energy_kwh = 0.0
    last_recorded_progress = 0
    
    for i, time in enumerate(times):
        moment_str = time.strftime('%Y-%m-%d %H:%M')
        
        # 获取太阳角度
        solar_data = angle_handler.getAngle(moment_str)
        apparent_elevation = solar_data['apparent_elevation'].iloc[0]
        solar_azimuth = solar_data['azimuth'].iloc[0]
        solar_zenith_deg = 90 - apparent_elevation
        
        # 获取基础辐射数据
        dni = solar_irradiance.get_dni(moment_str)
        ghi = solar_irradiance.get_ghi(moment_str)
        dhi = solar_irradiance.get_dhi(moment_str)
        
        # 计算大气层外法向辐射 dni_extra (Perez 模型必需)
        # 将 moment_str 转换为 datetime 对象
        moment_dt = pd.to_datetime(moment_str)
        dni_extra = pvlib.irradiance.get_extra_radiation(moment_dt)
        
        # 太阳在地平线以下
        if apparent_elevation <= 0:
            all_results.append({
                'moment': moment_str,
                'apparent_elevation': apparent_elevation,
                'total_power': 0.0,
                'face_powers': [0.0] * n_faces,
                'face_TIs': [0.0] * n_faces,
                'face_aois': [90.0] * n_faces
            })
            continue
        
        sun_above_count += 1
        
        # 计算每个面的辐照度
        face_TIs = []
        face_powers = []
        face_aois = []
        total_power = 0.0
        
        for face_azimuth in face_azimuths:
            # 计算入射角
            aoi = pvlib.irradiance.aoi(
                surface_tilt=surface_tilt,
                surface_azimuth=face_azimuth,
                solar_zenith=solar_zenith_deg,
                solar_azimuth=solar_azimuth
            )
            face_aois.append(aoi)
            
            # Perez 模型计算倾斜面辐照度 (添加 dni_extra)
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=surface_tilt,
                surface_azimuth=face_azimuth,
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                dni_extra=dni_extra,
                solar_zenith=solar_zenith_deg,
                solar_azimuth=solar_azimuth,
                albedo=albedo,
                model='perez'
            )
            
            # 如果 AOI > 90°，直射分量为 0
            if aoi >= 90:
                ti = poa['poa_sky_diffuse'] + poa['poa_ground_diffuse']
            else:
                ti = poa['poa_global']
            
            face_TIs.append(ti)
            face_power = ti * face_area
            face_powers.append(face_power)
            total_power += face_power
        
        # 梯形法权重
        weight = 0.5 if (i == 0 or i == len(times) - 1) else 1.0
        
        # 累加能量
        energy_wh = total_power * interval_hours * weight
        total_energy += energy_wh
        cumulative_energy_kwh += energy_wh / 1000
        
        for j in range(n_faces):
            face_energy[j] += face_powers[j] * interval_hours * weight
            face_valid_points[j] += weight
        
        all_results.append({
            'moment': moment_str,
            'apparent_elevation': apparent_elevation,
            'solar_azimuth': solar_azimuth,
            'solar_zenith_deg': solar_zenith_deg,
            'dni': dni,
            'ghi': ghi,
            'dhi': dhi,
            'dni_extra': dni_extra,
            'total_power': total_power,
            'face_TIs': face_TIs,
            'face_powers': face_powers,
            'face_aois': face_aois
        })
        
        # 进度记录 (每10%)
        current_progress = ((i + 1) / total_points * 100)
        if current_progress >= last_recorded_progress + 10 or i == len(times) - 1:
            progress_energy.append(cumulative_energy_kwh)
            last_recorded_progress = (int(current_progress / 10)) * 10
        
        # 打印进度
        if (i + 1) % 10 == 0 or i == len(times) - 1:
            print(f"  已处理: {i+1}/{len(times)} 个时刻")
    
    # 计算平均辐照度
    face_avg_irradiance = np.zeros(n_faces)
    for j in range(n_faces):
        if face_valid_points[j] > 0:
            face_avg_irradiance[j] = face_energy[j] / (face_valid_points[j] * interval_hours)
    
    total_area = n_faces * face_area
    if np.sum(face_valid_points) > 0:
        avg_valid_points = np.mean(face_valid_points)
        avg_irradiance = total_energy / (total_area * avg_valid_points * interval_hours)
    else:
        avg_irradiance = 0.0
    
    total_valid_time = face_valid_points[0] * interval_hours if n_faces > 0 else 0
    
    # 补充进度数据到10个点
    while len(progress_energy) < 10:
        progress_energy.append(progress_energy[-1] if progress_energy else 0.0)
    progress_energy = progress_energy[:10]
    
    # ========== 输出结果 ==========
    print("\n=== 汇总结果 ===")
    print(f"有效采样点数: {sun_above_count}/{len(times)} (太阳在地平线以上)")
    print(f"有效总时间 (单面): {total_valid_time:.2f} 小时")
    
    print(f"\n=== 各面累计结果 (每面面积 {face_area} m²) ===")
    for j in range(n_faces):
        print(f"  面 {j+1} (朝向 {face_azimuths[j]:.1f}°):")
        print(f"    累计能量: {face_energy[j]:.2f} Wh")
        print(f"    平均辐照度: {face_avg_irradiance[j]:.2f} W/m²")
        print(f"    有效时间: {face_valid_points[j] * interval_hours:.2f} 小时")
    
    print(f"\n=== 组件总累计结果 (所有 {n_faces} 面合计) ===")
    print(f"总累计能量: {total_energy:.2f} Wh")
    print(f"总累计能量: {total_energy/1000:.4f} kWh")
    print(f"加权平均辐照度 (所有面): {avg_irradiance:.2f} W/m²")
    
    # 打印首末时刻详细信息
    if all_results:
        print(f"\n=== 首末时刻详细结果示例 ===")
        for idx, label in enumerate(['首个时刻', '末个时刻']):
            r = all_results[0 if idx == 0 else -1]
            print(f"\n  --- {label}: {r['moment']} ---")
            print(f"  太阳高度角: {r['apparent_elevation']:.2f}°")
            print(f"  总功率: {r['total_power']:.2f} W")
            for j in range(n_faces):
                print(f"    面 {j+1} (朝向 {face_azimuths[j]:.1f}°): "
                      f"AOI={r['face_aois'][j]:.1f}°, "
                      f"TI={r['face_TIs'][j]:.2f} W/m², "
                      f"功率={r['face_powers'][j]:.2f} W")
    
    return {
        'total_energy_wh': total_energy,
        'total_energy_kwh': total_energy / 1000,
        'face_energy_wh': face_energy,
        'face_avg_irradiance': face_avg_irradiance,
        'face_valid_time': face_valid_points * interval_hours,
        'avg_irradiance': avg_irradiance,
        'sun_above_count': sun_above_count,
        'total_valid_time': total_valid_time,
        'all_results': all_results,
        'progress_energy': progress_energy,
        'face_azimuths': face_azimuths,
        'n_faces': n_faces
    }


# ========== 主程序：Perez 模型多面扫描 + 绘图 ==========
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Perez Model: Multi-face Scan (1 to 10 faces)")
    print("=" * 60)

    start_time = '2025-1-1 00:00'
    end_time = '2026-1-1 00:00'
    interval_hours = 4
    base_angle = 60
    albedo = 0.85  # 南极冰雪高反照率

    print(f"\n{'Faces':<6} {'Total Energy (kWh)':<20} {'Avg Irradiance (W/m²)':<25}")
    print("-" * 60)

    # 存储数据用于绘图
    face_counts = []
    avg_irradiances = []
    total_energies = []
    progress_data = {}

    for n in range(1, 11):
        result_range = calculate_prism_power_perez(
            latitude=-69.367,
            longitude=76.367,
            timezone='Asia/Shanghai',
            start_time=start_time,
            end_time=end_time,
            n_faces=n,
            base_angle=base_angle,
            face_area=1.0,
            surface_tilt=90,
            albedo=albedo,
            linke_turbidity=2.0,
            interval_hours=interval_hours
        )
        
        if result_range:
            print(f"{n:<6} {result_range['total_energy_kwh']:<20.4f} {result_range['avg_irradiance']:<25.2f}")
            face_counts.append(n)
            avg_irradiances.append(result_range['avg_irradiance'])
            total_energies.append(result_range['total_energy_kwh'])
            
            if 'progress_energy' in result_range:
                progress_data[n] = result_range['progress_energy']

    # ========== 图1: 绘制双轴柱状图 (平均辐照度 + 总能量) ==========
    if face_counts and avg_irradiances and total_energies:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Left axis: Avg Irradiance (bar chart)
        bars = ax1.bar(face_counts, avg_irradiances, color='steelblue', 
                       edgecolor='black', alpha=0.8, label='Avg Irradiance (Perez)')
        ax1.set_xlabel('Number of Faces (n)', fontsize=12)
        ax1.set_ylabel('Weighted Avg Irradiance (W/m²)', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Display values above bars
        for bar, val in zip(bars, avg_irradiances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='steelblue')
        
        # Right axis: Total Energy (line chart)
        ax2 = ax1.twinx()
        ax2.plot(face_counts, total_energies, color='coral', marker='o', 
                 linewidth=2, markersize=8, label='Total Energy (Perez)')
        ax2.set_ylabel('Total Energy (kWh)', fontsize=12, color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')
        
        # Display values on line points
        for x, val in zip(face_counts, total_energies):
            ax2.text(x, val + 50, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='coral')
        
        # Title with base_angle and albedo info
        plt.title(f'Perez Model: Number of Prism Faces vs Avg Irradiance and Total Energy\n'
                  f'Time Range: {start_time} to {end_time}, Interval: {interval_hours}h, '
                  f'Base Angle: {base_angle}°, Albedo: {albedo}', 
                  fontsize=14)
        plt.xticks(face_counts)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'perez_avg_irradiance_and_energy_base{base_angle}_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        print(f"\nFigure 1 saved as: {filename}")
        plt.show()
    else:
        print("No data available for plotting")

    # ========== 图2: 区间平均辐照度曲线 ==========
    if progress_data:
        fig2, ax3 = plt.subplots(figsize=(12, 7))
        
        progress_labels = [f'{i*10}%' for i in range(1, 11)]
        x_positions = np.arange(1, 11)
        colors = plt.cm.viridis(np.linspace(0, 1, len(progress_data)))
        
        for idx, (n, energies) in enumerate(sorted(progress_data.items())):
            if len(energies) < 2:
                continue
            
            if len(energies) > 10:
                energies = energies[:10]
            
            # 计算区间增量能量
            incremental_energies = []
            prev = 0.0
            for val in energies:
                incremental_energies.append(val - prev)
                prev = val
            
            while len(incremental_energies) < 10:
                incremental_energies.append(0.0)
            
            # 计算区间平均辐照度 (每个面的平均)
            total_valid_hours = result_range['total_valid_time'] if 'result_range' in locals() else 8760
            interval_hours_per_segment = total_valid_hours / 10
            
            avg_irradiance_per_interval = []
            for inc_energy in incremental_energies[:10]:
                if interval_hours_per_segment > 0:
                    avg_irr = (inc_energy * 1000) / (n * interval_hours_per_segment)
                else:
                    avg_irr = 0.0
                avg_irradiance_per_interval.append(avg_irr)
            
            ax3.plot(x_positions, avg_irradiance_per_interval, 
                    marker='o', linewidth=2, markersize=5, 
                    color=colors[idx], label=f'n={n}')
            
            max_idx = np.argmax(avg_irradiance_per_interval)
            ax3.text(x_positions[max_idx] + 0.1, avg_irradiance_per_interval[max_idx] + 1, 
                    f'{avg_irradiance_per_interval[max_idx]:.1f}', 
                    fontsize=7, ha='left', va='bottom')
        
        ax3.set_xlabel('Progress (Time Intervals)', fontsize=12)
        ax3.set_ylabel('Average Irradiance per Face (W/m²)', fontsize=12)
        ax3.set_title(f'Perez Model: Average Irradiance per Face Across Time Intervals\n'
                      f'Time Range: {start_time} to {end_time}, Base Angle: {base_angle}°, '
                      f'Albedo: {albedo}, Interval: {interval_hours}h', 
                      fontsize=14)
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(progress_labels)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend(loc='upper left', title='Number of Faces', fontsize=9)
        
        plt.tight_layout()
        
        timestamp2 = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename2 = f'perez_avg_irradiance_distribution_base{base_angle}_{timestamp2}.png'
        plt.savefig(filename2, dpi=150)
        print(f"\nFigure 2 saved as: {filename2}")
        plt.show()
    else:
        print("No progress data available for plotting")