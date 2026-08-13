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
        
        参数:
            moment: 时间字符串, 格式 'YYYY-MM-DD HH:MM', 如 '2025-06-22 9:00'
            alpha_ws: 白天空反照率 (从BRDF数据获取)
            alpha_bs: 黑天空反照率 (从BRDF数据获取, 需匹配当前太阳天顶角)
            
        返回:
            dict: 包含所有面详细结果和总功率
        """
        # --> 获取太阳角度 elevation  azimuth
        solar_data = self.angle_handler.getAngle(moment)
        apparent_elevation = solar_data['apparent_elevation'].iloc[0]
        solar_azimuth = solar_data['azimuth'].iloc[0]
        
        # 如果太阳在地平线以下，所有面功率为0
        if apparent_elevation <= 0:
            return {
                'moment': moment,
                'latitude': self.latitude,
                'longitude': self.longitude,
                'apparent_elevation': apparent_elevation,   # 添加这个字段
                'solar_azimuth': solar_azimuth,             # 添加这个字段
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
                        'sun_above_horizon': False
                    }
                    for i, face_azimuth in enumerate(self.face_azimuths)
                ]
            }
        
        # --> 获取基础辐射数据 dni ghi dhi
        dni = self.solar_irradiance.get_dni(moment)
        ghi = self.solar_irradiance.get_ghi(moment)
        dhi = self.solar_irradiance.get_dhi(moment)
        
        # --> 计算每个面的辐照度 
        face_results = []
        total_power = 0.0
        total_irradiance_weighted = 0.0
        total_area = self.n_faces * self.face_area
        
        for i, face_azimuth in enumerate(self.face_azimuths):
            # 使用 AngleCombination 计算该面的入射角 AOI
            aoi_deg = self.angle_handler.AngleCombination(moment, face_azimuth)
            
            # 调用 SingleFaceIrradiance 计算该面辐照度
            result = self.face_calculator.calculate(
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                aoi_deg=aoi_deg,
                apparent_elevation=apparent_elevation,
                alpha_ws=alpha_ws,
                alpha_bs=alpha_bs
            )
            
            # 计算该面功率 (W)
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
            'face_results': face_results,
            'total_power': total_power,
            'total_area': total_area,
            'weighted_avg_irradiance': total_irradiance_weighted / total_area
        }

# python3 -m IrradianceModule.TI_Handler
if __name__ == "__main__":
    # 创建组件功率计算类
    prism_power = PrismPowerCalculator(
        latitude=-69.367,
        longitude=76.367,
        timezone='Asia/Shanghai',
        n_faces=3,
        base_angle=0,
        face_area=1.0,
        linke_turbidity=2.0
    )
    
    # 计算单个时刻
    test_time = '2025-12-22 12:00'
    alpha_ws = 0.85
    alpha_bs = 0.75
    
    result = prism_power.calculate_power(test_time, alpha_ws, alpha_bs)
    
    print(f"=== 测试时间: {test_time} ===")
    print(f"太阳高度角: {result['apparent_elevation']:.2f}°")
    print(f"太阳方位角: {result['solar_azimuth']:.2f}°")
    print(f"\n棱柱各面结果 (共 {len(result['face_results'])} 面, 每面 {prism_power.face_area} m²):")
    
    for face in result['face_results']:
        status = "☀️" if face['sun_above_horizon'] else "🌙"
        print(f"  面 {face['face_index']} (朝向 {face['face_azimuth']:.1f}°): "
              f"AOI = {face['aoi_deg']:.1f}°, "
              f"TI = {face['TI']:.2f} W/m², "
              f"功率 = {face['power']:.2f} W {status}")
    
    print(f"\n总功率: {result['total_power']:.2f} W")
    print(f"加权平均辐照度: {result['weighted_avg_irradiance']:.2f} W/m²")