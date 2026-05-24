from AngleHandler import AngleHandler
from IrradianceHandler import SolarIrradiance
from ComponentsHandler import AngleGenerator
import numpy as np
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pvlib
import calendar

def getSynthesisEfficiency(latitude, longitude, timeZone, year, month, day, hour, minute, face, baseAngle):
    demoAngleHandler = AngleHandler(latitude, longitude, timeZone)
    dayTime = demoAngleHandler.timeTransfer(year, month, day, hour, minute)

    # get棱柱各面角度
    demoComponentsHandler = AngleGenerator(face, baseAngle)
    array = demoComponentsHandler.generate()

    # get直射辐照度 DNI
    demoIrradianceHandler = SolarIrradiance(latitude, longitude, timeZone)
    DNI = demoIrradianceHandler.get_dni(dayTime)

    # 首先判断太阳是否在地平线以下
    if demoAngleHandler.AngleCombination(dayTime, array[0]) == -1000:
        return 0
    else:
        result = 0
        for i in array:
            if demoAngleHandler.is_within_90(dayTime, i):
                angle_deg = demoAngleHandler.AngleCombination(dayTime, i)
                if angle_deg != -1000:
                    # 转弧度再计算 cos
                    angle_rad = np.radians(angle_deg)
                    result += DNI * np.cos(angle_rad)

        # get平均辐照度
        result = result / face 
        return result

def getPowerObtainWithin_24h(latitude, longitude, timeZone, year, month, day, face, baseAngle):
    result = 0
    for i in range(0, 24):      # 0~23 点
        for j in range(0, 60):  # 0~59 分
            result += getSynthesisEfficiency(latitude, longitude, timeZone, year, month, day, i, j, face, baseAngle) * 60
    return result / 3600000  # 单位：KWh/m²

def getPowerObtainWith_in_24h(latitude, longitude, timeZone, year, month, day, face, baseAngle):
    demoAngleHandler = AngleHandler(latitude, longitude, timeZone)
    demoIrradianceHandler = SolarIrradiance(latitude, longitude, timeZone)

    # 生成全天每分钟时间序列
    times = []
    for h in range(0, 24):
        for m in range(0, 60):
            times.append(demoAngleHandler.timeTransfer(year, month, day, h, m))

    # 一次性计算太阳位置和 DNI
    solar_positions = [demoAngleHandler.getAngle(t) for t in times]
    DNIs = [demoIrradianceHandler.get_dni(t) for t in times]

    # 生成棱柱各面角度
    demoComponentsHandler = AngleGenerator(face, baseAngle)
    array = demoComponentsHandler.generate()

    # 遍历每分钟和每个面计算辐照度
    total_energy = 0
    for idx, t in enumerate(times):
        apparent_elevation = solar_positions[idx]['apparent_elevation'].iloc[0]
        azimuth = solar_positions[idx]['azimuth'].iloc[0]
        DNI = DNIs[idx]

        if apparent_elevation <= 0 or DNI <= 0:
            continue  # 太阳在地平线以下，跳过

        minute_energy = 0
        for angle in array:
            # 判断 ±90° 范围
            diff = (azimuth - angle + 180) % 360 - 180
            if abs(diff) <= 90:
                # 计算法向入射角
                elev_rad = np.radians(apparent_elevation)
                az_diff_rad = np.radians(diff)
                cos_theta = np.cos(elev_rad) * np.cos(az_diff_rad)
                cos_theta = max(cos_theta, 0)  # 避免负值
                minute_energy += DNI * cos_theta

        total_energy += minute_energy / face * 60  # 乘60秒，平均到每面

    return total_energy / 3600000  # 转为 kWh/m²

def getPowerObtainWithinMonth(latitude, longitude, timeZone, year, month,
                              face, baseAngle, step_minutes=10, albedo=0.85):
    demoAngleHandler = AngleHandler(latitude, longitude, timeZone)
    demoIrradianceHandler = SolarIrradiance(latitude, longitude, timeZone)

    import calendar
    days_in_month = calendar.monthrange(year, month)[1]

    # 生成棱柱各面角度
    demoComponentsHandler = AngleGenerator(face, baseAngle)
    array = demoComponentsHandler.generate()

    total_energy = 0.0

    for day in range(1, days_in_month + 1):
        # 生成当天每 step minutes 时间序列
        times = []
        for h in range(0, 24):
            for m in range(0, 60, step_minutes):
                times.append(demoAngleHandler.timeTransfer(year, month, day, h, m))

        # 一次性计算太阳位置和 DNI/GHI/DHI
        solar_positions = [demoAngleHandler.getAngle(t) for t in times]
        DNIs = [demoIrradianceHandler.get_dni(t) for t in times]
        GHIs = [demoIrradianceHandler.get_ghi(t) for t in times]
        DHIs = [demoIrradianceHandler.get_dhi(t) for t in times]

        for idx, t in enumerate(times):
            apparent_elevation = solar_positions[idx]['apparent_elevation'].iloc[0]
            azimuth = solar_positions[idx]['azimuth'].iloc[0]
            DNI = DNIs[idx]
            GHI = GHIs[idx]
            DHI = DHIs[idx]

            if apparent_elevation <= 0 or (DNI <= 0 and GHI <= 0):
                continue

            minute_energy = 0.0
            for angle in array:
                diff = (azimuth - angle + 180) % 360 - 180
                if abs(diff) <= 90:
                    # 法向入射角
                    elev_rad = np.radians(apparent_elevation)
                    az_diff_rad = np.radians(diff)
                    cos_theta = np.cos(elev_rad) * np.cos(az_diff_rad)
                    cos_theta = max(cos_theta, 0)

                    # 入射角 AOI
                    aoi_deg = np.degrees(np.arccos(cos_theta)) if cos_theta > 0 else 90

                    # 直射
                    direct = DNI * cos_theta

                    # 散射 + 地面反射
                    diffuse = demoIrradianceHandler.hay_davies_diffuse(
                        DHI=DHI, DNI=DNI, GHI=GHI,
                        solar_zenith_deg=90 - apparent_elevation,
                        aoi_deg=aoi_deg, tilt_deg=90,
                        albedo=albedo
                    )

                    minute_energy += direct + diffuse

            # 每个时间点对应 step minutes 分钟
            total_energy += minute_energy / face * step_minutes * 60

    return total_energy / 3600000.0  # kWh/m²

def getPowerObtainWithinYear(latitude, longitude, timeZone, year,
                             face, baseAngle, step_minutes=10, albedo=0.85):
    demoAngleHandler = AngleHandler(latitude, longitude, timeZone)
    demoIrradianceHandler = SolarIrradiance(latitude, longitude, timeZone)

    import calendar

    # 生成棱柱各面角度
    demoComponentsHandler = AngleGenerator(face, baseAngle)
    array = demoComponentsHandler.generate()

    total_energy = 0.0

    # 遍历12个月
    for month in range(1, 13):
        days_in_month = calendar.monthrange(year, month)[1]

        for day in range(1, days_in_month + 1):
            # 生成当天每 step_minutes 时间序列
            times = []
            for h in range(0, 24):
                for m in range(0, 60, step_minutes):
                    times.append(demoAngleHandler.timeTransfer(year, month, day, h, m))

            # 一次性计算太阳位置和 DNI/GHI/DHI
            solar_positions = [demoAngleHandler.getAngle(t) for t in times]
            DNIs = [demoIrradianceHandler.get_dni(t) for t in times]
            GHIs = [demoIrradianceHandler.get_ghi(t) for t in times]
            DHIs = [demoIrradianceHandler.get_dhi(t) for t in times]

            for idx, t in enumerate(times):
                apparent_elevation = solar_positions[idx]['apparent_elevation'].iloc[0]
                azimuth = solar_positions[idx]['azimuth'].iloc[0]
                DNI = DNIs[idx]
                GHI = GHIs[idx]
                DHI = DHIs[idx]

                if apparent_elevation <= 0 or (DNI <= 0 and GHI <= 0):
                    continue

                minute_energy = 0.0
                for angle in array:
                    diff = (azimuth - angle + 180) % 360 - 180
                    if abs(diff) <= 90:
                        # 法向入射角
                        elev_rad = np.radians(apparent_elevation)
                        az_diff_rad = np.radians(diff)
                        cos_theta = np.cos(elev_rad) * np.cos(az_diff_rad)
                        cos_theta = max(cos_theta, 0)

                        # 入射角 AOI
                        aoi_deg = np.degrees(np.arccos(cos_theta)) if cos_theta > 0 else 90

                        # 直射
                        direct = DNI * cos_theta

                        # 散射 + 地面反射
                        diffuse = demoIrradianceHandler.hay_davies_diffuse(
                            DHI=DHI, DNI=DNI, GHI=GHI,
                            solar_zenith_deg=90 - apparent_elevation,
                            aoi_deg=aoi_deg, tilt_deg=90,
                            albedo=albedo
                        )

                        minute_energy += direct + diffuse

                # 每个时间点对应 step_minutes 分钟
                total_energy += minute_energy / face * step_minutes * 60

    return total_energy / 3600000.0  # kWh/m²

def getPowerObtainWithinYear_plus(latitude, longitude, timeZone, year,
                             face, baseAngle, step_minutes=10, albedo=0.85):
    demoAngleHandler = AngleHandler(latitude, longitude, timeZone)
    demoIrradianceHandler = SolarIrradiance(latitude, longitude, timeZone)
    demoComponentsHandler = AngleGenerator(face, baseAngle)
    array = demoComponentsHandler.generate()

    # -------------------------------
    # 生成整年时间序列
    # -------------------------------
    times = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:59',
                          freq=f'{step_minutes}min', tz=timeZone)

    # 批量计算太阳位置
    location = pvlib.location.Location(latitude, longitude, tz=timeZone)
    solpos = location.get_solarposition(times)
    apparent_elevations = solpos['apparent_elevation'].values
    azimuths = solpos['azimuth'].values

    # 批量计算 clearsky
    clearsky = location.get_clearsky(times, model='ineichen', linke_turbidity=3)
    DNIs = clearsky['dni'].values
    GHIs = clearsky['ghi'].values
    DHIs = clearsky['dhi'].values

    total_energy = 0.0

    # -------------------------------
    # 定义计算单面辐照度的函数
    # -------------------------------
    def calc_panel_energy(args):
        apparent_elevation, azimuth, DNI, GHI, DHI, angle = args
        if apparent_elevation <= 0 or (DNI <= 0 and GHI <= 0):
            return 0.0
        diff = (azimuth - angle + 180) % 360 - 180
        if abs(diff) > 90:
            return 0.0
        elev_rad = np.radians(apparent_elevation)
        az_diff_rad = np.radians(diff)
        cos_theta = np.cos(elev_rad) * np.cos(az_diff_rad)
        cos_theta = max(cos_theta, 0.0)
        aoi_deg = np.degrees(np.arccos(cos_theta)) if cos_theta > 0 else 90
        direct = DNI * cos_theta
        diffuse = demoIrradianceHandler.hay_davies_diffuse(
            DHI=DHI, DNI=DNI, GHI=GHI,
            solar_zenith_deg=90 - apparent_elevation,
            aoi_deg=aoi_deg, tilt_deg=90,
            albedo=albedo
        )
        return direct + diffuse

    # -------------------------------
    # 遍历时间点，使用线程池计算每个面
    # -------------------------------
    for idx in range(len(times)):
        args_list = [(apparent_elevations[idx], azimuths[idx],
                      DNIs[idx], GHIs[idx], DHIs[idx], angle)
                     for angle in array]

        with ThreadPoolExecutor() as executor:
            panel_energies = list(executor.map(calc_panel_energy, args_list))

        # step_minutes 对应的秒数
        minute_energy = sum(panel_energies) / face
        total_energy += minute_energy * step_minutes * 60

    return total_energy / 3600000.0  # 转 kWh/m²



if __name__ == '__main__':
    # result = getSynthesisEfficiency(-69.367, 76.367, 'Etc/GMT-5', 2025, 3, 20, 14, 0, 3, 0)
    # print(result)

    # result1 = getPowerObtainWithinMonth(-69.367, 76.367, 'Etc/GMT-5', 2025, 12,  3, 0, 30, 0.85)
    # print(result1)

    print('1 face: ')
    print(getPowerObtainWithinYear_plus(-69.367, 76.367, 'Etc/GMT-5', 2025,   1, 0, 60, 0.85))
    print('\n2 face: ')
    print(getPowerObtainWithinYear_plus(-69.367, 76.367, 'Etc/GMT-5', 2025, 2, 0, 60, 0.85))
    print('\n3 face: ')
    print(getPowerObtainWithinYear_plus(-69.367, 76.367, 'Etc/GMT-5', 2025, 3, 0, 60, 0.85))
    print('\n4 face: ')
    print(getPowerObtainWithinYear_plus(-69.367, 76.367, 'Etc/GMT-5', 2025, 4, 0, 60, 0.85))
    print('\n5 face: ')
    print(getPowerObtainWithinYear_plus(-69.367, 76.367, 'Etc/GMT-5', 2025, 5, 0, 60, 0.85))
    print('\n6 face: ')
    print(getPowerObtainWithinYear_plus(-69.367, 76.367, 'Etc/GMT-5', 2025, 6, 0, 60, 0.85))
    print('\n7 face: ')
    print(getPowerObtainWithinYear_plus(-69.367, 76.367, 'Etc/GMT-5', 2025,   10, 0, 60, 0.85))
    print('\n10 face: ')

    result_year_1 = getPowerObtainWithinYear_plus(-69.367, 76.367, 'Etc/GMT-5', 2025,   30, 0, 60, 0.85)
    print(result_year_1)

    # for i in range(0, 121, 10):
    #     result1 = getPowerObtainWithinMonth(-69.367, 76.367, 'Etc/GMT-5', 2025, 7,  3, i)
    #     print(result1)
    #     print('\n')

