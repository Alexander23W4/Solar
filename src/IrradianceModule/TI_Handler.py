import numpy as np
import pandas as pd
import pvlib

from AngleModule.AngleHandler import AngleHandler
from IrradianceModule.IrradianceHandler import SolarIrradiance
from AssemblyModule.ComponentsHandler import AngleGenerator

class SingleFaceIrradiance:
    """
    单面倾斜面辐照度计算类
    
    所有角度作为外部输入，不依赖任何角度计算模块。
    仅根据输入的太阳角度、面板朝向和基础辐射数据计算倾斜面总辐照度。
    
    公式:
    TI(t) = DNI(t) * cos(AOI) * fIAM(AOI) 
          + DHI(t) * fIAM_diff * ((1+cosβ)/2) 
          + GHI(t) * αeff′(t) * ((1-cosβ)/2) * fIAM_diff

    IAM = (1 - exp(-cos(AOI) / a_r)) / (1 - exp(-1 / a_r))
    AOI：入射角，即太阳光线与光伏板法线（即板面的垂直线）的夹角,单位为度。
    a_r：角度损失系数，它是一个经验值，用于描述玻璃的光学特性
    
    其中:
    αeff′(t) = [kdiff(t) * αws(t) + (1−kdiff(t)) * αbs(θSun, t)] * 0.98 * flow_sun(γ)
    """
    
    def __init__(self, surface_tilt=90):
        """
        参数:
            surface_tilt: 光伏板倾角 β (度), 默认90° (垂直)
            surface_azimuth: 光伏板方位角 γ (度), 默认0° (正北)
                             以正北为0°，顺时针增加
        """
        self.beta = np.radians(surface_tilt)           # 倾角 (弧度)
        self.surface_tilt_deg = surface_tilt
        
        # ============ 可调参数 ============
        self.fIAM_diff = 0.986           # 散射光的IAM修正因子 (典型值0.9-1.0)
    
    def _calculate_fIAM_direct(self, aoi_deg):
        """
        计算直射光的 FIAM (入射角修正因子)
        使用 Martin & Ruiz 单参数模型: IAM = (1 - exp(-cos(AOI) / a_r)) / (1 - exp(-1 / a_r))
        
        参数:
            aoi_deg: 入射角 (度)
            a_r: 角度损失系数, 文献推荐值 0.163 (用于非抗反射涂层玻璃)
            
        返回:
            fIAM: 直射光IAM修正因子 (0~1)
        """
        a_r = 0.163
        
        # 入射角超过90度, 光线从背面入射, 修正因子为0
        if aoi_deg >= 90:
            return 0.0
        
        # 将入射角转为弧度
        aoi_rad = np.radians(aoi_deg)
        
        # Martin & Ruiz 模型公式
        # IAM(θ) = (1 - exp(-cos(θ) / a_r)) / (1 - exp(-1 / a_r))
        cos_aoi = np.cos(aoi_rad)
        
        # 防止数值溢出 (cos_aoi 在 [0, 1] 范围, 不会溢出, 但保持健壮性)
        numerator = 1 - np.exp(-cos_aoi / a_r)
        denominator = 1 - np.exp(-1 / a_r)
        
        fIAM = numerator / denominator
        
        # 确保结果在合理范围内
        # 理论上 AOI=0 时 IAM=1, AOI=90 时 IAM=0
        return np.clip(fIAM, 0.0, 1.0)
    
    def _calculate_alpha_eff(self, dhi, ghi, apparent_elevation, alpha_ws, alpha_bs):
        """
        计算有效反照率 αeff′(t)
        
        公式: αeff′ = [kdiff * αws + (1−kdiff) * αbs] * 0.98 * flow_sun
        
        参数:
            dhi: 水平散射辐照度 (W/m²)
            ghi: 水平总辐照度 (W/m²)
            apparent_elevation: 视太阳高度角 (度), 用于 flow_sun 计算
            alpha_ws: 白天空反照率 (外部输入, 从BRDF数据获取)
            alpha_bs: 黑天空反照率 (外部输入, 从BRDF数据获取, 需匹配太阳天顶角)
            
        返回:
            alpha_eff: 有效反照率
        """
        # 计算直射/散射比例因子 kdiff = DHI/GHI 
        if ghi > 0:
            kdiff = dhi / ghi
            kdiff = np.clip(kdiff, 0.0, 1.0)  # 确保在合理范围
        else:
            kdiff = 1.0  # 无阳光时全为散射
        
        # 低角度修正因子 flow_sun  (当太阳高度角低于10°时, BRDF模型失效, 需要平滑压制地面反射)
        flow_sun = self._calculate_flow_sun(apparent_elevation)
        
        # αeff′ = [kdiff * αws + (1−kdiff) * αbs] * 0.98 * flow_sun
        alpha_eff = (kdiff * alpha_ws + (1 - kdiff) * alpha_bs) * 0.98 * flow_sun
        
        return np.clip(alpha_eff, 0.0, 1.0)

    def _calculate_flow_sun(self, apparent_elevation):
        """
        计算低角度修正因子 flow_sun(γ)
        
        当太阳高度角低于10°时, BRDF模型失效且散射比例突变, 需要平滑压制地面反射分量。
        
        参数:
            apparent_elevation: 视太阳高度角 (度)
        
        返回:
            flow_sun: 修正因子 (0~1)
        """
        threshold = 10.0  # 阈值 (度), 基于文献
        p = 1.8           # 平滑指数 (1.5~2.0)
        
        if apparent_elevation >= threshold:
            return 1.0
        elif apparent_elevation <= 0:
            return 0.0
        else:
            # 从阈值到0进行平滑过渡
            ratio = apparent_elevation / threshold
            return ratio ** p
        
    def calculate(self, dni, ghi, dhi, aoi_deg, apparent_elevation, alpha_ws, alpha_bs):
        """
        计算单个倾斜面的总辐照度 (所有角度作为外部输入)
        
        参数:
            dni: 法向直接辐照度 (W/m²)
            ghi: 水平总辐照度 (W/m²)
            dhi: 水平散射辐照度 (W/m²)
            aoi_deg: 入射角 (度), 太阳光线与光伏板法线的夹角
            apparent_elevation: 视太阳高度角 (度), 用于判断昼夜和 flow_sun 计算
            alpha_ws: 白天空反照率 (从BRDF数据获取)
            alpha_bs: 黑天空反照率 (从BRDF数据获取, 需匹配当前太阳天顶角)
            
        返回:
            dict: 包含 TI 及各分量的详细结果
        """
        # 判断太阳是否在地平线以上 
        if apparent_elevation <= 0:
            return {
                'TI': 0.0,
                'direct_component': 0.0,
                'sky_diffuse_component': 0.0,
                'ground_reflected_component': 0.0,
                'alpha_eff': 0.0,
                'aoi_deg': aoi_deg,
                'fIAM_direct': 0.0,
                'fIAM_diff': self.fIAM_diff,
                'sun_above_horizon': False
            }
        
        # 计算各修正因子 FIAM FIAM_diff
        fIAM_direct = self._calculate_fIAM_direct(aoi_deg)
        fIAM_diff = self.fIAM_diff
        
        # 计算 alpha_eff (传入 alpha_ws 和 alpha_bs)
        alpha_eff = self._calculate_alpha_eff(dhi, ghi, apparent_elevation, alpha_ws, alpha_bs)
        
        # 入射角弧度
        aoi_rad = np.radians(aoi_deg)
        
        # 直接辐射分量: DNI * cos(AOI) * fIAM(AOI)
        direct_component = dni * np.cos(aoi_rad) * fIAM_direct
        
        # 天空散射分量: DHI * fIAM_diff * ((1+cosβ)/2)
        sky_diffuse_component = dhi * fIAM_diff * (1 + np.cos(self.beta)) / 2
        
        # 地面反射分量: GHI * αeff′ * ((1-cosβ)/2) * fIAM_diff
        ground_reflected_component = ghi * alpha_eff * (1 - np.cos(self.beta)) / 2 * fIAM_diff
        
        # 总辐照度
        TI = direct_component + sky_diffuse_component + ground_reflected_component
        
        # 返回详细结果
        return {
            'TI': TI,
            'direct_component': direct_component,
            'sky_diffuse_component': sky_diffuse_component,
            'ground_reflected_component': ground_reflected_component,
            'alpha_eff': alpha_eff,
            'aoi_deg': aoi_deg,
            'fIAM_direct': fIAM_direct,
            'fIAM_diff': fIAM_diff,
            'sun_above_horizon': True
        }


# PrismPowerCalculator这个class的需求是, 输入地理位置和一个时刻, n棱柱, base_angle, alpha_ws, alpha_bs以及其他调用类需要的参数, 
# 内部调用angle_handler angle_generator生成 每个面的AOI
# 调用solar_irradiance产生这个时刻这个地点的 DHI DNI GHI  
# 然后每个面往里面往SingleFaceIrradiance一代入, 就把整个组件的辐照度功率算出来了

class PrismPowerCalculator:
    """
    棱柱形光伏组件辐照度功率计算类
    
    输入地理位置和时刻，内部调用：
    1. AngleHandler → 获取太阳角度和每个面的 AOI
    2. SolarIrradiance → 获取 DNI, GHI, DHI
    3. AngleGenerator → 生成各面朝向
    4. SingleFaceIrradiance → 计算每个面的辐照度
    
    汇总得到整个棱柱组件的总功率。
    """
    
    def __init__(self, latitude, longitude, timezone, 
                 n_faces=3, base_angle=0, face_area=1.0,
                 linke_turbidity=2.0):
        """
        参数:
            latitude: 纬度 (度)
            longitude: 经度 (度)
            timezone: 时区, 如 'Asia/Shanghai'
            n_faces: 棱柱面数, 默认3 (三棱柱)
            base_angle: 第一个面的朝向角度 (度), 以正北为0°, 顺时针增加
            face_area: 每个面的面积 (m²), 默认1.0
            linke_turbidity: 林克浑浊度, 用于TBIE计算, 默认2.0 
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.face_area = face_area
        
        # 初始化调用类: 
        self.angle_handler = AngleHandler(
            latitude=latitude,
            longitude=longitude,
            timeZone=timezone
        )
        
        self.solar_irradiance = SolarIrradiance(
            latitude=latitude,
            longitude=longitude,
            tz=timezone,
            linke_turbidity=linke_turbidity
        )
        
        self.angle_generator = AngleGenerator(n=n_faces, base_angle=base_angle)
        self.face_azimuths = self.angle_generator.generate()
        self.n_faces = len(self.face_azimuths)
        
        self.face_calculator = SingleFaceIrradiance(surface_tilt=90)
    
    def calculate_power(self, moment, alpha_ws, alpha_bs):
        """
        计算单个时刻棱柱组件所有面的辐照度和总功率
        """
        # --> 获取太阳角度 elevation  azimuth
        solar_data = self.angle_handler.getAngle(moment)
        apparent_elevation = solar_data['apparent_elevation'].iloc[0]
        solar_azimuth = solar_data['azimuth'].iloc[0]
        
        # 如果太阳在地平线以下
        if apparent_elevation <= 0:
            return {
                'moment': moment,
                'latitude': self.latitude,
                'longitude': self.longitude,
                'apparent_elevation': apparent_elevation,   
                'solar_azimuth': solar_azimuth,
                'dni': 0.0,
                'ghi': 0.0,
                'dhi': 0.0,
                'kdiff': 1.0,
                'alpha_ws': alpha_ws,
                'alpha_bs': alpha_bs,
                'total_power': 0.0,
                'total_area': self.n_faces * self.face_area,
                'weighted_avg_irradiance': 0.0,
                'face_results': [
                    {
                        'face_index': i + 1,
                        'face_azimuth': face_azimuth,
                        'aoi_deg': 90.0,
                        'TI': 0.0,
                        'power': 0.0,
                        'fIAM_direct': 0.0,
                        'alpha_eff': 0.0,
                        'sun_above_horizon': False
                    }
                    for i, face_azimuth in enumerate(self.face_azimuths)
                ]
            }
        
        # --> 获取基础辐射数据
        dni = self.solar_irradiance.get_dni(moment)
        ghi = self.solar_irradiance.get_ghi(moment)
        dhi = self.solar_irradiance.get_dhi(moment)
        
        # 计算 kdiff
        if ghi > 0:
            kdiff = dhi / ghi
            kdiff = np.clip(kdiff, 0.0, 1.0)
        else:
            kdiff = 1.0
        
        # --> 计算每个面的辐照度 
        face_results = []
        total_power = 0.0
        total_irradiance_weighted = 0.0
        total_area = self.n_faces * self.face_area
        
        for i, face_azimuth in enumerate(self.face_azimuths):
            aoi_deg = self.angle_handler.AngleCombination(moment, face_azimuth)
            
            result = self.face_calculator.calculate(
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                aoi_deg=aoi_deg,
                apparent_elevation=apparent_elevation,
                alpha_ws=alpha_ws,
                alpha_bs=alpha_bs
            )
            
            face_power = result['TI'] * self.face_area
            total_power += face_power
            total_irradiance_weighted += result['TI'] * self.face_area
            
            face_results.append({
                'face_index': i + 1,
                'face_azimuth': face_azimuth,
                'aoi_deg': aoi_deg,
                'TI': result['TI'],
                'power': face_power,
                'fIAM_direct': result['fIAM_direct'],
                'alpha_eff': result['alpha_eff'],
                'sun_above_horizon': result['sun_above_horizon']
            })
        
        return {
            'moment': moment,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'apparent_elevation': apparent_elevation,
            'solar_azimuth': solar_azimuth,
            'dni': dni,
            'ghi': ghi,
            'dhi': dhi,
            'kdiff': kdiff,
            'alpha_ws': alpha_ws,
            'alpha_bs': alpha_bs,
            'face_results': face_results,
            'total_power': total_power,
            'total_area': total_area,
            'weighted_avg_irradiance': total_irradiance_weighted / total_area
        }



# 月份		 αws
# 1月		0.81
# 2月		0.83
# 3月		0.85
# 4月		0.87
# 5月		0.88
# 6月		0.89
# 7月		0.89
# 8月		0.88
# 9月		0.87
# 10月		0.85
# 11月		0.83
# 12月		0.82

# αbs(θSun)=αws+0.08×max(0,cos(θSun)−0.3)

 # 新增：月份到 alpha_ws 的查找表
    def _get_alpha_ws_by_month(self, month):
        """
        根据月份返回 alpha_ws (白天空反照率)
        
        参数:
            month: 月份 (1-12)
        
        返回:
            alpha_ws: 该月的白天空反照率
        """
        month_aws = {
            1: 0.81, 2: 0.83, 3: 0.85, 4: 0.87,
            5: 0.88, 6: 0.89, 7: 0.89, 8: 0.88,
            9: 0.87, 10: 0.85, 11: 0.83, 12: 0.82
        }
        return month_aws.get(month, 0.85)
    
    # 新增：根据 alpha_ws 和太阳天顶角计算 alpha_bs
    def _calculate_alpha_bs(self, alpha_ws, solar_zenith_deg):
        """
        根据 alpha_ws 和太阳天顶角计算 alpha_bs (黑天空反照率)
        
        公式: alpha_bs(θSun) = alpha_ws + 0.08 * max(0, cos(θSun) - 0.3)
        
        参数:
            alpha_ws: 白天空反照率
            solar_zenith_deg: 太阳天顶角 (度)
        
        返回:
            alpha_bs: 黑天空反照率
        """
        solar_zenith_rad = np.radians(solar_zenith_deg)
        cos_zenith = np.cos(solar_zenith_rad)
        correction = 0.08 * max(0, cos_zenith - 0.3)
        return alpha_ws + correction
    
    # 新增：时间范围伪积分函数
    def calculate_power_time_range(self, start_time, end_time, interval_hours=1):
        """
        计算指定时间范围内的总辐照度 (伪积分)
        
        参数:
            start_time: 开始时间字符串, 格式 'YYYY-MM-DD HH:MM'
            end_time: 结束时间字符串, 格式 'YYYY-MM-DD HH:MM'
            interval_hours: 时间间隔 (小时), 默认1小时
        
        返回:
            dict: 包含总能量和各面的累计结果
        """
        # 生成时间序列
        times = pd.date_range(
            start=start_time, 
            end=end_time, 
            freq=f'{interval_hours}h', 
            tz=self.timezone
        )
        
        if len(times) == 0:
            print("错误: 时间范围无效")
            return None
        
        # 存储每个时刻的结果
        all_results = []
        
        print(f"=== 开始计算时间范围: {start_time} 到 {end_time} ===")
        print(f"时间间隔: {interval_hours} 小时, 共 {len(times)} 个采样点\n")
        
        for i, time in enumerate(times):
            moment_str = time.strftime('%Y-%m-%d %H:%M')
            
            # 提取月份, 自动获取 alpha_ws
            month = time.month
            alpha_ws = self._get_alpha_ws_by_month(month)
            
            # 获取太阳角度来计算 alpha_bs
            solar_data = self.angle_handler.getAngle(moment_str)
            apparent_elevation = solar_data['apparent_elevation'].iloc[0]
            solar_zenith_deg = 90 - apparent_elevation
            
            # 计算 alpha_bs
            alpha_bs = self._calculate_alpha_bs(alpha_ws, solar_zenith_deg)
            
            # 计算该时刻的功率
            result = self.calculate_power(moment_str, alpha_ws=alpha_ws, alpha_bs=alpha_bs)
            all_results.append(result)
            
            # 打印进度
            if (i + 1) % 10 == 0 or i == len(times) - 1:
                print(f"  已处理: {i+1}/{len(times)} 个时刻")
        
        print("\n=== 伪积分汇总 ===")
        
        # 初始化累计变量
        face_energy = np.zeros(self.n_faces)          # 每个面的累计能量 (Wh)
        face_valid_points = np.zeros(self.n_faces)    # 每个面的有效时间点数 (加权)
        total_energy = 0.0                            # 所有面总能量 (Wh)
        sun_above_count = 0                           # 太阳在地平线以上的采样点数
        
        # 遍历所有结果进行积分
        for i, result in enumerate(all_results):
            # 判断太阳是否在地平线以上 (只看一个面即可，所有面共用同一太阳位置)
            if result['apparent_elevation'] > 0:
                sun_above_count += 1
                
                # 梯形法权重: 首尾点权重0.5，中间点权重1.0
                weight = 0.5 if (i == 0 or i == len(all_results) - 1) else 1.0
                
                # 累加总能量
                total_energy += result['total_power'] * interval_hours * weight
                
                # 累加每个面的能量和有效时间点数
                for j, face in enumerate(result['face_results']):
                    face_energy[j] += face['power'] * interval_hours * weight
                    face_valid_points[j] += weight
        
        # ========== 计算每个面的平均辐照度 ==========
        face_avg_irradiance = np.zeros(self.n_faces)
        for j in range(self.n_faces):
            if face_valid_points[j] > 0:
                face_avg_irradiance[j] = face_energy[j] / (face_valid_points[j] * interval_hours)
            else:
                face_avg_irradiance[j] = 0.0
        
        # ========== 计算整体加权平均辐照度 ==========
        # 方法: 所有面总能量 / (所有面总面积 * 平均有效时间)
        total_area = self.n_faces * self.face_area
        if np.sum(face_valid_points) > 0:
            avg_valid_points = np.mean(face_valid_points)  # 所有面的平均有效时间点数
            avg_irradiance = total_energy / (total_area * avg_valid_points * interval_hours)
        else:
            avg_irradiance = 0.0
        
        # 计算总有效时间 (用于显示，取任意一个面的有效时间即可)
        total_valid_time = face_valid_points[0] * interval_hours if self.n_faces > 0 else 0
        
        # ========== 输出结果 ==========
        print(f"\n=== 地理位置 ===")
        print(f"纬度: {self.latitude:.3f}°")
        print(f"经度: {self.longitude:.3f}°")
        
        print(f"\n=== 时间范围 ===")
        print(f"开始: {start_time}")
        print(f"结束: {end_time}")
        print(f"时间间隔: {interval_hours} 小时")
        print(f"有效采样点数: {sun_above_count}/{len(times)} (太阳在地平线以上)")
        print(f"有效总时间 (单面): {total_valid_time:.2f} 小时")
        
        print(f"\n=== 各面累计结果 (每面面积 {self.face_area} m²) ===")
        for j in range(self.n_faces):
            face_azimuth = self.face_azimuths[j]
            print(f"\n  面 {j+1} (朝向 {face_azimuth:.1f}°):")
            print(f"    累计能量: {face_energy[j]:.2f} Wh")
            print(f"    平均辐照度: {face_avg_irradiance[j]:.2f} W/m²")
            print(f"    有效时间: {face_valid_points[j] * interval_hours:.2f} 小时")
        
        print(f"\n=== 组件总累计结果 (所有 {self.n_faces} 面合计) ===")
        print(f"总累计能量: {total_energy:.2f} Wh")
        print(f"总累计能量: {total_energy/1000:.4f} kWh")
        print(f"加权平均辐照度 (所有面): {avg_irradiance:.2f} W/m²")
        
        # 打印首个和末个时刻的详细信息
        print(f"\n=== 首末时刻详细结果示例 ===")
        for idx, moment_label in enumerate(['首个时刻', '末个时刻']):
            result_idx = 0 if idx == 0 else -1
            result = all_results[result_idx]
            
            print(f"\n  --- {moment_label}: {result['moment']} ---")
            print(f"  太阳高度角: {result['apparent_elevation']:.2f}°")
            print(f"  太阳方位角: {result['solar_azimuth']:.2f}°")
            print(f"  kdiff: {result['kdiff']:.4f}")
            print(f"  alpha_ws: {result['alpha_ws']:.3f}")
            print(f"  alpha_bs: {result['alpha_bs']:.3f}")
            print(f"  总功率: {result['total_power']:.2f} W")
            
            for face in result['face_results']:
                status = "☀️" if face['sun_above_horizon'] else "🌙"
                print(f"    面 {face['face_index']} (朝向 {face['face_azimuth']:.1f}°): "
                      f"AOI={face['aoi_deg']:.1f}°, "
                      f"TI={face['TI']:.2f} W/m², "
                      f"功率={face['power']:.2f} W {status}")
        
        return {
            'total_energy_wh': total_energy,
            'total_energy_kwh': total_energy / 1000,
            'face_energy_wh': face_energy,
            'face_avg_irradiance': face_avg_irradiance,
            'face_valid_time': face_valid_points * interval_hours,
            'avg_irradiance': avg_irradiance,
            'sun_above_count': sun_above_count,
            'total_valid_time': total_valid_time,
            'all_results': all_results
        }

# python3 -m IrradianceModule.TI_Handler
if __name__ == "__main__":
    linke_turbidity = 2.0
    prism_power = PrismPowerCalculator(
        latitude=-69.367,
        longitude=76.367,
        timezone='Asia/Shanghai',
        n_faces=1,
        base_angle=0,
        face_area=1.0,
        linke_turbidity=linke_turbidity
    )
    
    # # ========== 测试1: 单时刻计算 ==========
    # print("=" * 60)
    # print("测试1: 单时刻计算")
    # print("=" * 60)
    
    # test_time = '2025-12-22 12:00'
    
    # # 获取该时刻的 alpha_ws 和 alpha_bs
    # month = pd.to_datetime(test_time).month
    # alpha_ws = prism_power._get_alpha_ws_by_month(month)
    # solar_data = prism_power.angle_handler.getAngle(test_time)
    # apparent_elevation = solar_data['apparent_elevation'].iloc[0]
    # solar_zenith_deg = 90 - apparent_elevation
    # alpha_bs = prism_power._calculate_alpha_bs(alpha_ws, solar_zenith_deg)
    
    # result_single = prism_power.calculate_power(test_time, alpha_ws, alpha_bs)
    
    # print(f"=== 地理位置 ===")
    # print(f"纬度: {result_single['latitude']:.3f}°")
    # print(f"经度: {result_single['longitude']:.3f}°")
    # print(f"林克浑浊度 (Linke Turbidity): {linke_turbidity}")
    
    # print(f"\n=== 测试时刻: {test_time} ===")
    # print(f"太阳高度角: {result_single['apparent_elevation']:.2f}°")
    # print(f"太阳方位角: {result_single['solar_azimuth']:.2f}°")
    
    # print(f"\n=== 辐射参数 ===")
    # print(f"DNI: {result_single['dni']:.2f} W/m²")
    # print(f"GHI: {result_single['ghi']:.2f} W/m²")
    # print(f"DHI: {result_single['dhi']:.2f} W/m²")
    # print(f"kdiff (DHI/GHI): {result_single['kdiff']:.4f}")
    # print(f"alpha_ws (白天空反照率): {result_single['alpha_ws']:.3f}")
    # print(f"alpha_bs (黑天空反照率): {result_single['alpha_bs']:.3f}")
    
    # print(f"\n=== 棱柱各面结果 (共 {len(result_single['face_results'])} 面, 每面 {prism_power.face_area} m²) ===")
    
    # for face in result_single['face_results']:
    #     status = "☀️" if face['sun_above_horizon'] else "🌙"
    #     print(f"\n  面 {face['face_index']} (朝向 {face['face_azimuth']:.1f}°):")
    #     print(f"    AOI           = {face['aoi_deg']:.1f}°")
    #     print(f"    fIAM_direct   = {face['fIAM_direct']:.4f}")
    #     print(f"    alpha_eff     = {face['alpha_eff']:.4f}")
    #     print(f"    TI            = {face['TI']:.2f} W/m²")
    #     print(f"    功率          = {face['power']:.2f} W {status}")
    
    # print(f"\n=== 汇总 ===")
    # print(f"总功率: {result_single['total_power']:.2f} W")
    # print(f"加权平均辐照度: {result_single['weighted_avg_irradiance']:.2f} W/m²")
    
    # # ========== 测试2: 时间范围伪积分 (新增) ==========
    # print("\n" + "=" * 60)
    # print("测试2: 时间范围伪积分")
    # print("=" * 60)
    
    # start_time = '2025-1-22 12:00'
    # end_time = '2026-1-22 12:00'
    # interval_hours = 12
    
    # result_range = prism_power.calculate_power_time_range(
    #     start_time=start_time,
    #     end_time=end_time,
    #     interval_hours=interval_hours
    # )
    
    # if result_range:
    #     print(f"\n=== 伪积分最终汇总 ===")
    #     print(f"总累计能量: {result_range['total_energy_kwh']:.4f} kWh")
    #     print(f"加权平均辐照度: {result_range['avg_irradiance']:.2f} W/m²")

    # ========== 测试3: 多面扫描 (1-10面) ==========
    import matplotlib.pyplot as plt
    from datetime import datetime

    print("\n" + "=" * 60)
    print("Test 3: Multi-face Scan (1 to 10 faces)")
    print("=" * 60)

    start_time = '2025-1-1 00:00'
    end_time = '2026-1-1 00:00'
    interval_hours = 4

    print(f"\n{'Faces':<6} {'Total Energy (kWh)':<20} {'Avg Irradiance (W/m²)':<25}")
    print("-" * 60)

    # 存储数据用于绘图
    face_counts = []
    avg_irradiances = []
    total_energies = []

    for n in range(1, 11):
        prism_power = PrismPowerCalculator(
            latitude=-69.367,
            longitude=76.367,
            timezone='Asia/Shanghai',
            n_faces=n,
            base_angle=0,
            face_area=1.0,
            linke_turbidity=2.0
        )
        
        result_range = prism_power.calculate_power_time_range(
            start_time=start_time,
            end_time=end_time,
            interval_hours=interval_hours
        )
        
        if result_range:
            print(f"{n:<6} {result_range['total_energy_kwh']:<20.4f} {result_range['avg_irradiance']:<25.2f}")
            face_counts.append(n)
            avg_irradiances.append(result_range['avg_irradiance'])
            total_energies.append(result_range['total_energy_kwh'])

    # ========== 绘制双轴柱状图 (平均辐照度 + 总能量) ==========
    if face_counts and avg_irradiances and total_energies:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Left axis: Avg Irradiance (bar chart)
        bars = ax1.bar(face_counts, avg_irradiances, color='steelblue', 
                    edgecolor='black', alpha=0.8, label='Avg Irradiance')
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
        line = ax2.plot(face_counts, total_energies, color='coral', marker='o', 
                        linewidth=2, markersize=8, label='Total Energy')
        ax2.set_ylabel('Total Energy (kWh)', fontsize=12, color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')
        
        # Display values on line points
        for x, val in zip(face_counts, total_energies):
            ax2.text(x, val + 50, f'{val:.0f}', ha='center', va='bottom', fontsize=8, color='coral')
        
        plt.title(f'Number of Prism Faces vs Avg Irradiance and Total Energy\nTime Range: {start_time} to {end_time}, Interval: {interval_hours}h', fontsize=14)
        plt.xticks(face_counts)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # ========== Save figure with timestamp ==========
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prism_avg_irradiance_and_energy_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        print(f"\nFigure saved as: {filename}")
        
        # Display figure
        plt.show()
    else:
        print("No data available for plotting")

