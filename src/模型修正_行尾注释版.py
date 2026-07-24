import pvlib  # 光伏系统建模库，用于太阳位置计算、辐照量计算等
import pandas as pd  # 数据处理与表格操作
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 数据可视化
import seaborn as sns  # 统计图表美化
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)  # 设置随机种子，确保每次运行结果可复现

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "PingFang SC"]  # 配置中文字体，按优先级尝试 SimHei、Microsoft YaHei、PingFang SC
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

STATIONS = {
    "长城站": {
        "latitude": -62.1265,  # 纬度（南纬为负）
        "longitude": -58.9904,  # 经度（西经为负）
        "altitude": 10,  # 海拔高度（米）
        "temperature": -2.8,  # 年平均气温（℃）
        "pressure": 980,  # 大气压（hPa）
        "timezone": 'Etc/GMT+3',  # 时区（Etc/GMT+3 表示西三区）

        "P_DC0": 500,  # 组件额定直流功率（W）
        "M": -70,  # Marion模型中的温度-辐照耦合系数
        "SC_THIN": 0.35,  # 薄雪层积雪系数（雪厚<3cm时）
        "SC_THICK": 0.05,  # 厚雪层积雪系数（雪厚≥3cm时）
        "TILT_DEFAULT": 90,  # 默认组件倾角（°），用于气象模拟
        "GAMMA": -0.0042,  # 功率温度系数（%/℃）
        "NOCT": 40,  # 标称工作电池温度（℃）
        "MODULE_HEIGHT": 170,  # 组件高度（cm）
        "MODULE_CLEARANCE": 50,  # 组件底部离地高度（cm）
        "IAM_BO": 0.035,  # 入射角修正系数（ASHRAE模型参数）

        "snow_albedo": 0.85,  # 纯雪面反照率
        "bare_albedo": 0.2,  # 裸地反照率
        "base_snow_depth": 5,  # 基础积雪深度（cm）
        "max_snow_depth": 80,  # 最大积雪深度（cm）
        "snow_melt_temp": 0,  # 积雪融化温度阈值（℃）

        "snow_density_dry": 0.2,  # 干雪密度（g/cm³）
        "snow_density_wet": 0.45,  # 湿雪密度（g/cm³）
        "snow_density_ice": 0.85,  # 冰壳密度（g/cm³）
        "snow_density_max": 0.4,  # 干雪最大密度（g/cm³）
        "freeze_thaw_temp": -2,  # 冻融循环温度阈值（℃）
        "ice_crust_coeff": 0.2,  # 冰壳积雪系数修正因子
        "wet_snow_coeff": 1.5,  # 湿雪积雪系数修正因子
        "dry_snow_coeff": 0.8,  # 干雪积雪系数修正因子
    },
    "中山站": {
        "latitude": -69.3714,  # 纬度（南纬为负），比长城站更靠近南极点
        "longitude": 76.3700,  # 经度（东经为正）
        "altitude": 11,
        "temperature": -10.0,  # 年平均气温（℃），显著低于长城站
        "pressure": 970,  # 大气压（hPa）
        "timezone": 'Etc/GMT-7',  # 时区（Etc/GMT-7 表示东七区）

        "P_DC0": 500,  # 组件额定直流功率（W）
        "M": -75,
        "SC_THIN": 0.30,  # 薄雪层积雪系数（雪厚<3cm时）
        "SC_THICK": 0.04,  # 厚雪层积雪系数（雪厚≥3cm时）
        "TILT_DEFAULT": 90,  # 默认组件倾角（°），用于气象模拟
        "GAMMA": -0.0040,  # 功率温度系数（%/℃）
        "NOCT": 38,  # 标称工作电池温度（℃）
        "MODULE_HEIGHT": 170,  # 组件高度（cm）
        "MODULE_CLEARANCE": 50,  # 组件底部离地高度（cm）
        "IAM_BO": 0.04,  # 入射角修正系数（ASHRAE模型参数）

        "snow_albedo": 0.88,  # 纯雪面反照率
        "bare_albedo": 0.2,  # 裸地反照率
        "base_snow_depth": 20,  # 基础积雪深度（cm）
        "max_snow_depth": 50,  # 最大积雪深度（cm）
        "snow_melt_temp": 1,  # 积雪融化温度阈值（℃）

        "snow_density_dry": 0.15,  # 干雪密度（g/cm³）
        "snow_density_wet": 0.45,  # 湿雪密度（g/cm³）
        "snow_density_ice": 0.85,  # 冰壳密度（g/cm³）
        "snow_density_max": 0.35,  # 干雪最大密度（g/cm³）
        "freeze_thaw_temp": -3,  # 冻融循环温度阈值（℃）
        "ice_crust_coeff": 0.15,  # 冰壳积雪系数修正因子
        "wet_snow_coeff": 1.4,  # 湿雪积雪系数修正因子
        "dry_snow_coeff": 0.7,  # 干雪积雪系数修正因子
    }
}


def validate_input(prompt, min_val, max_val, input_type=float, default=None):  # 用户输入验证函数，确保输入在有效范围内
    while True:  # 循环直到输入有效
        try:
            user_input = input(prompt + (f" [默认: {default}]: " if default else ": "))  # 获取用户输入，如有默认值则显示
            if not user_input and default is not None:  # 用户直接回车且存在默认值时，返回默认值
                return default
            value = input_type(user_input)  # 转换输入为指定类型
            if min_val <= value <= max_val:  # 检查数值是否在有效范围内
                return value
            else:
                print(f"输入错误！请输入{min_val}到{max_val}之间的数值。")
        except ValueError:  # 类型转换失败时的错误处理
            print(f"输入错误！请输入有效的{input_type.__name__}类型数值。")


def simulate_antarctic_meteorology(station_data, year=2023):  # 模拟南极科考站全年逐小时气象数据
    times = pd.date_range(  # 创建全年逐小时的时间索引，带时区信息
        start=f'{year}-01-01 00:00:00',
        end=f'{year}-12-31 23:00:00',
        freq='h',  # 每小时一个数据点
        tz=station_data["timezone"]
    )
    n_hours = len(times)  # 全年总小时数（通常为8760或8784）
    hour_idx = np.arange(n_hours)  # 小时索引数组 [0, 1, 2, ...]
    hour_of_day = hour_idx % 24  # 一天中的小时 [0-23]
    day_of_year = times.dayofyear.values  # 一年中的第几天 [1-365]

    solpos = pvlib.solarposition.get_solarposition(  # 使用pvlib计算每个时刻的太阳高度角、方位角等
        times,
        latitude=station_data["latitude"],
        longitude=station_data["longitude"],
        altitude=station_data["altitude"],
        temperature=station_data["temperature"],
        pressure=station_data["pressure"]
    )

    zenith = np.deg2rad(solpos['apparent_zenith'].values)  # 天顶角（太阳与正上方的夹角），转换为弧度
    sin_alt = np.sin(np.deg2rad(90) - zenith)  # sin(高度角) = cos(天顶角)，即太阳高度角的正弦值
    sin_alt[sin_alt < 0] = 0  # 夜间太阳在地平线以下，sin_alt为负，将其置为0
    day_mask = sin_alt > 0  # 白天掩码：太阳在地平线以上时为True

    tilt_default = station_data["TILT_DEFAULT"]  # 默认90°（垂直安装）
    azimuth_default = 0  # 正北方向

    aoi = pvlib.irradiance.aoi(  # 计算太阳光线与组件法线之间的夹角
        tilt_default, azimuth_default,
        solpos['apparent_zenith'], solpos['azimuth']
    )
    aoi_rad = np.deg2rad(aoi.values)  # 转换为弧度
    cos_aoi = np.cos(aoi_rad)  # 入射角余弦值
    cos_aoi[cos_aoi < 0] = 0  # 背面入射时置为0
    iam = np.where(
        aoi_rad < np.deg2rad(90),
        1 - station_data["IAM_BO"] * (1 / np.cos(np.clip(aoi_rad, 0, np.deg2rad(89))) - 1),  # IAM = 1 - IAM_BO * (1/cos(AOI) - 1)
        0
    )
    iam = np.clip(iam, 0, 1)  # 限制在[0,1]范围内

    ghi_clear = np.zeros(n_hours)  # 初始化晴空水平面总辐照数组
    ghi_clear[day_mask] = 850 * sin_alt[day_mask]  # 简化模型：GHI_clear = 850 * sin(高度角)，850为近似大气层顶辐照

    direct_frac = np.zeros(n_hours)  # 初始化直射分量占比数组
    direct_frac[day_mask] = 0.6 + 0.3 * sin_alt[day_mask]  # 直射占比随太阳高度增加而增大（0.6~0.9）
    ghi_b_clear = ghi_clear * direct_frac  # 晴空直射辐照
    ghi_d_clear = ghi_clear * (1 - direct_frac)  # 晴空散射辐照
    dni_clear = np.zeros(n_hours)
    dni_clear[day_mask] = ghi_b_clear[day_mask] / sin_alt[day_mask]  # 直射法向辐照(DNI) = 直射水平辐照 / sin(高度角)

    cloud_factor = 0.35 + 0.5 * np.random.rand(n_hours)  # 随机生成云量因子（0.35~0.85之间随机），模拟云层遮挡
    cloud_factor = np.clip(cloud_factor, 0.20, 0.80)  # 限制在合理范围

    ghi = ghi_clear * cloud_factor  # 实际辐照量 = 晴空辐照 * 云量因子
    dni = dni_clear * cloud_factor
    ghi_d = ghi_d_clear * cloud_factor
    ghi[~day_mask] = 0  # 夜间辐照置为0
    dni[~day_mask] = 0
    ghi_d[~day_mask] = 0

    tilt_rad = np.deg2rad(tilt_default)  # 默认倾角转弧度
    poa_direct = dni * cos_aoi * iam  # 直射分量：DNI * cos(AOI) * IAM
    poa_sky_diffuse = ghi_d * (1 + np.cos(tilt_rad)) / 2  # 天空散射分量：使用各向同性模型
    poa_ground_diffuse = ghi * station_data["snow_albedo"] * (1 - np.cos(tilt_rad)) / 2  # 地面反射分量：GHI * 反照率 * (1 - cos(倾角)) / 2
    poa_total = poa_direct + poa_sky_diffuse + poa_ground_diffuse  # 总倾斜面辐照 = 直射 + 散射 + 反射
    poa_total = np.clip(poa_total, 0, 1200)  # 限制最大值
    poa_total[~day_mask] = 0  # 夜间置为0

    seasonal_cycle = np.cos(2 * np.pi * (day_of_year - 15) / 365)  # 季节性温度变化：余弦函数模拟年周期

    if station_data["latitude"] == -69.3714:
        base_temp = -10 + 12 * seasonal_cycle  # 中山站基础温度：更冷，年温差更大
    else:
        base_temp = -2.5 + 7.5 * seasonal_cycle  # 长城站基础温度

    temp_diurnal = np.where(  # 日温度变化：白天有显著日变化，夜间变化较小
        sin_alt > 0,
        3 * np.sin(2 * np.pi * (hour_of_day - 12) / 24),  # 白天振幅3℃
        0.5 * np.sin(2 * np.pi * (hour_of_day - 12) / 24)  # 夜间振幅0.5℃
    )
    temp_amb = base_temp + temp_diurnal + np.random.normal(0, 1.2, n_hours)  # 叠加随机噪声模拟天气波动
    temp_amb = np.clip(temp_amb, -40, 8)  # 限制在合理范围

    snow_depth = np.full(n_hours, station_data["base_snow_depth"])  # 初始化积雪深度为基础值

    if station_data["latitude"] == -69.3714:
        monthly_snow = [
            (1, 5, 20), (1, 15, 22), (1, 25, 20),
            (2, 5, 23), (2, 15, 25), (2, 25, 22),
            (3, 5, 27), (3, 15, 30), (3, 25, 28),
            (4, 5, 33), (4, 15, 36), (4, 25, 34),
            (5, 5, 38), (5, 15, 40), (5, 25, 39),
            (6, 5, 41), (6, 15, 42), (6, 25, 41),
            (7, 5, 43), (7, 15, 44), (7, 25, 43),
            (8, 5, 45), (8, 15, 46), (8, 25, 45),
            (9, 5, 47), (9, 15, 48), (9, 25, 46),
            (10, 5, 44), (10, 15, 41), (10, 25, 38),
            (11, 5, 33), (11, 15, 30), (11, 25, 27),
            (12, 5, 24), (12, 15, 22), (12, 25, 21)
        ]
    else:
        monthly_snow = [
            (1, 5, 8), (1, 15, 7), (1, 25, 6),
            (2, 5, 9), (2, 15, 10), (2, 25, 9),
            (3, 5, 14), (3, 15, 17), (3, 25, 15),
            (4, 5, 24), (4, 15, 28), (4, 25, 26),
            (5, 5, 38), (5, 15, 45), (5, 25, 42),
            (6, 5, 52), (6, 15, 58), (6, 25, 55),
            (7, 5, 63), (7, 15, 68), (7, 25, 66),
            (8, 5, 70), (8, 15, 73), (8, 25, 71),
            (9, 5, 68), (9, 15, 62), (9, 25, 57),
            (10, 5, 47), (10, 15, 42), (10, 25, 37),
            (11, 5, 28), (11, 15, 23), (11, 25, 19),
            (12, 5, 12), (12, 15, 10), (12, 25, 9)
        ]

    snow_events = []  # 存储积雪事件的时间索引范围
    for month, day, max_depth in monthly_snow:
        try:
            start_date = pd.Timestamp(year=year, month=month, day=day, tz=station_data["timezone"])  # 创建事件开始时间戳
            start_idx = int((start_date - times[0]).total_seconds() / 3600)  # 计算相对于年初的小时索引
            if 0 <= start_idx < n_hours - 72:
                snow_events.append((start_idx, start_idx + 72, max_depth))  # 事件持续72小时
        except:
            continue

    for start, end, max_depth in snow_events:
        if start < n_hours and end < n_hours:
            snow_depth[start:end] = np.linspace(  # 积雪事件期间，积雪深度线性增加到最大值
                snow_depth[start-1] if start > 0 else station_data["base_snow_depth"],
                max_depth,
                end - start
            )

    for i in range(1, n_hours):
        in_snow_event = any([start <= i < end for start, end, _ in snow_events])  # 检查当前是否在积雪事件期间
        if not in_snow_event:  # 非积雪事件期间，积雪深度保持不变
            snow_depth[i] = snow_depth[i-1]

        if sin_alt[i] > 0 and temp_amb[i] > station_data["snow_melt_temp"]:
            melt_rate = 0.08 * temp_amb[i] + 0.0002 * poa_total[i]  # 融化速率与温度和辐照正相关
            snow_depth[i] = max(station_data["base_snow_depth"] * 0.3, snow_depth[i] - melt_rate)  # 积雪深度减少，但不低于基础值的30%

    snow_depth = np.clip(snow_depth, 0, station_data["max_snow_depth"])  # 限制积雪深度在合理范围

    snow_density = np.zeros(n_hours)  # 雪密度数组
    snow_state = np.full(n_hours, "干雪")  # 雪状态数组（干雪/湿雪/融雪/冰壳/无雪）
    freeze_thaw_temp = station_data["freeze_thaw_temp"]
    melt_temp = station_data["snow_melt_temp"]
    current_density = station_data["snow_density_dry"]  # 当前密度（随时间演变）

    for i in range(n_hours):
        if snow_depth[i] < 0.01:  # 无雪情况
            snow_density[i] = 0
            snow_state[i] = "无雪"  # 标记为无雪
            current_density = station_data["snow_density_dry"]  # 当前密度（随时间演变）
            continue

        if temp_amb[i] < -3:  # 冰壳判断：温度低于-3℃
            window_start = max(0, i - 12)
            melt_hours = np.sum(temp_amb[window_start:i] > melt_temp)  # 过去12小时内有>=6小时高于融化温度则形成冰壳
            if melt_hours >= 6:
                snow_state[i] = "冰壳"  # 标记为冰壳状态
                snow_density[i] = station_data["snow_density_ice"]
                continue

        if temp_amb[i] >= melt_temp:  # 温度高于融化阈值：融雪状态
            snow_state[i] = "融雪"  # 标记为融雪状态
            snow_density[i] = min(station_data["snow_density_wet"], 0.45)
            current_density = station_data["snow_density_wet"]

        elif freeze_thaw_temp <= temp_amb[i] < melt_temp:  # 温度在冻融区间：湿雪状态
            snow_state[i] = "湿雪"  # 标记为湿雪状态
            snow_density[i] = min(station_data["snow_density_wet"], 0.45)
            current_density = station_data["snow_density_wet"]

        else:
            snow_state[i] = "干雪"  # 标记为干雪状态
            current_density = min(station_data["snow_density_max"], current_density + 0.000015)  # 干雪密度缓慢增加（压实）
            snow_density[i] = current_density

    meteo_df = pd.DataFrame({  # 组装气象数据DataFrame
        "poa_total_Wm2": poa_total,  # 总倾斜面辐照量 (W/m2)
        "poa_direct_Wm2": poa_direct,  # 直射分量 (W/m2)
        "poa_sky_diffuse_Wm2": poa_sky_diffuse,  # 天空散射分量 (W/m2)
        "poa_ground_diffuse_Wm2": poa_ground_diffuse,  # 地面反射分量 (W/m2)
        "ghi_Wm2": ghi,  # 水平面总辐照 (W/m2)
        "temp_amb_C": temp_amb,  # 环境温度 (C)
        "snow_depth_cm": snow_depth,  # 积雪深度 (cm)
        "snow_density_gcm3": snow_density,  # 雪密度 (g/cm3)
        "snow_state": snow_state,  # 雪状态（字符串）
        "sin_alt": sin_alt,  # 太阳高度角正弦值
        "iam": iam,  # 入射角修正因子
        "zenith": zenith,  # 天顶角（弧度）
        "solar_zenith": solpos['apparent_zenith'].values,  # 太阳天顶角（度）
        "solar_azimuth": solpos['azimuth'].values,  # 太阳方位角（度）
        "hour_of_day": hour_of_day,  # 一天中的小时 [0-23]
        "day_of_year": day_of_year  # 一年中的第几天 [1-365]
    }, index=times)  # 以时间序列作为索引

    return meteo_df  # 返回气象数据DataFrame


def calculate_marion_model_power(station_data, meteo_df, tilt, azimuth):  # 基于Marion模型计算光伏组件功率输出，考虑积雪覆盖影响
    n_hours = len(meteo_df)  # 获取总小时数
    sin_alt = meteo_df["sin_alt"].values  # 从气象数据中提取相关序列
    day_mask = sin_alt > 0  # 白天掩码：太阳在地平线以上时为True
    ghi = meteo_df["ghi_Wm2"].values  # 水平面总辐照序列
    temp_amb = meteo_df["temp_amb_C"].values  # 环境温度序列
    snow_depth = meteo_df["snow_depth_cm"].values  # 积雪深度序列
    snow_state = meteo_df["snow_state"].values  # 雪状态序列
    solar_zenith = meteo_df["solar_zenith"].values  # 太阳天顶角序列
    solar_azimuth = meteo_df["solar_azimuth"].values  # 太阳方位角序列

    aoi = pvlib.irradiance.aoi(tilt, azimuth, solar_zenith, solar_azimuth)  # 计算太阳光线与组件法线之间的夹角
    aoi_rad = np.deg2rad(aoi)
    cos_aoi = np.cos(aoi_rad)  # 入射角余弦值
    cos_aoi[cos_aoi < 0] = 0  # 背面入射时置为0
    iam = np.where(
        aoi_rad < np.deg2rad(90),
        1 - station_data["IAM_BO"] * (1 / np.cos(np.clip(aoi_rad, 0, np.deg2rad(89))) - 1),  # IAM = 1 - IAM_BO * (1/cos(AOI) - 1)
        0
    )
    iam = np.clip(iam, 0, 1)  # 限制在[0,1]范围内

    direct_frac = np.zeros(n_hours)  # 初始化直射分量占比数组
    direct_frac[day_mask] = 0.6 + 0.3 * sin_alt[day_mask]  # 直射占比随太阳高度增加而增大（0.6~0.9）
    ghi_d = ghi * (1 - direct_frac)  # 散射分量
    dni = np.zeros(n_hours)
    dni[day_mask] = (ghi * direct_frac)[day_mask] / sin_alt[day_mask]  # 直射法向辐照

    tilt_rad = np.deg2rad(tilt)
    snow_cover_frac = np.clip(snow_depth / 10, 0, 1)  # 积雪覆盖比例（10cm为完全覆盖）
    albedo = station_data["bare_albedo"] + (station_data["snow_albedo"] - station_data["bare_albedo"]) * snow_cover_frac  # 反照率随积雪覆盖比例变化：积雪越多，反照率越高

    poa_direct = dni * cos_aoi * iam  # 直射分量：DNI * cos(AOI) * IAM
    poa_sky_diffuse = ghi_d * (1 + np.cos(tilt_rad)) / 2  # 天空散射分量：使用各向同性模型
    poa_ground_diffuse = ghi * albedo * (1 - np.cos(tilt_rad)) / 2
    poa_total = poa_direct + poa_sky_diffuse + poa_ground_diffuse  # 总倾斜面辐照 = 直射 + 散射 + 反射
    poa_total = np.clip(poa_total, 0, 1200)  # 限制最大值
    poa_total[~day_mask] = 0  # 夜间置为0

    temp_cell = temp_amb + (poa_total / 800) * (station_data["NOCT"] - 20)  # 电池温度 = 环境温度 + (POA/800) * (NOCT - 20)
    p_baseline = station_data["P_DC0"] * (poa_total / 1000) * (1 + station_data["GAMMA"] * (temp_cell - 25))  # 基准功率 = 额定功率 * (POA/1000) * (1 + 温度系数 * (电池温度 - 25))
    p_baseline = np.clip(p_baseline, 0, station_data["P_DC0"])  # 限制在额定功率范围内
    p_baseline[~day_mask] = 0  # 夜间功率置为0

    current_cover = 0.0  # 当前组件积雪覆盖比例 [0,1]
    f_snow = np.zeros(n_hours)  # 积雪覆盖比例序列
    p_actual = np.zeros(n_hours)  # 实际功率序列
    module_h = station_data["MODULE_HEIGHT"]  # 组件高度（cm）
    module_clearance = station_data["MODULE_CLEARANCE"]  # 底部离地高度（cm）
    module_top = module_h + module_clearance  # 组件顶部高度（cm）

    for i in range(n_hours):
        if snow_depth[i] >= module_top:  # 情况1：积雪完全覆盖组件
            current_cover = 1.0  # 完全覆盖
            f_snow[i] = 1.0  # 记录完全覆盖
            p_actual[i] = 0.0  # 完全遮挡，无发电
            continue

        if snow_depth[i] <= module_clearance:  # 情况2：积雪在组件底部以下
            base_cover = 0.0  # 无覆盖
        else:
            excess_snow = snow_depth[i] - module_clearance  # 超出底部的积雪
            base_cover = min(1.0, excess_snow / module_h)  # 基础覆盖比例

        if i > 0 and snow_depth[i] > snow_depth[i-1] + 0.3:  # 积雪显著增加时，额外覆盖比例
            snow_increase = snow_depth[i] - snow_depth[i-1]  # 积雪增量
            cover_increase = (snow_increase / module_h) * 0.35  # 额外覆盖比例
            current_cover = min(1.0, max(base_cover, current_cover + cover_increase))  # 更新覆盖比例
        else:
            current_cover = max(current_cover, base_cover)  # 取较大值

        snow_on_module = current_cover * module_h  # 计算组件上的积雪厚度

        sc_base = station_data["SC_THIN"] if snow_on_module < 3 else station_data["SC_THICK"]  # 根据积雪厚度选择积雪系数

        if snow_state[i] == "冰壳":  # 冰壳：更难滑落
            sc = sc_base * station_data["ice_crust_coeff"]  # 冰壳修正
        elif snow_state[i] in ("湿雪", "融雪"):  # 湿雪/融雪：更易滑落
            sc = sc_base * station_data["wet_snow_coeff"]  # 湿雪/融雪修正
        elif snow_state[i] == "干雪":  # 干雪：中等滑落
            sc = sc_base * station_data["dry_snow_coeff"]  # 干雪修正
        else:
            sc = sc_base

        sin_tilt = np.sin(np.deg2rad(tilt))  # 计算倾角因子
        if tilt > 60:  # 倾角>60度时，滑落因子增长趋缓
            sin_60 = np.sin(np.deg2rad(60))
            tilt_factor = sin_60 + (sin_tilt - sin_60) * 0.10
        else:
            tilt_factor = sin_tilt

        slide_condition = temp_amb[i] > (poa_total[i] / station_data["M"])  # 滑落条件：温度 > POA/M（Marion模型温度-辐照耦合条件）
        if slide_condition and sin_alt[i] > 0 and current_cover > 0:  # 且白天、有积雪覆盖
            slide_amount = sc * tilt_factor  # 滑落量
            current_cover = max(0.0, current_cover - slide_amount)  # 更新覆盖比例

        if current_cover > 0 and poa_total[i] > 0:  # 积雪融化机制（辐照加热）
            snow_thickness = current_cover * module_h  # 积雪厚度
            melt_thickness = poa_total[i] * 0.00018  # 辐照导致的融化
            snow_thickness = max(0.0, snow_thickness - melt_thickness)  # 融化后厚度
            current_cover = snow_thickness / module_h  # 更新覆盖比例

        f_snow[i] = current_cover  # 记录当前覆盖比例

        if snow_on_module < 2:  # 薄雪（<2cm）有一定透光性
            transmittance = 1.0 - (snow_on_module / 2.0)  # 透光率计算
        else:
            transmittance = 0.0  # 厚雪完全遮挡
        transmittance = np.clip(transmittance, 0.0, 1.0)  # 限制范围

        if current_cover <= 0:  # 无覆盖无损失
            loss_ratio = 0.0
        elif 0 < current_cover <= 1/3:  # 低覆盖：损失1/3
            loss_ratio = 1/3  # 损失比例1/3
        elif 1/3 < current_cover <= 2/3:  # 中覆盖：损失2/3
            loss_ratio = 2/3  # 损失比例2/3
        else:  # 高覆盖：完全损失
            loss_ratio = 1.0  # 完全损失

        effective_loss = loss_ratio * (1.0 - transmittance)  # 有效损失 = 损失比例 * (1 - 透光率)
        p_actual[i] = p_baseline[i] * (1.0 - effective_loss)  # 实际功率 = 基准功率 * (1 - 有效损失)
        p_actual[i] = np.clip(p_actual[i], 0.0, station_data["P_DC0"])  # 限制在额定功率范围内

    poa_components = pd.DataFrame({  # 组装辐照分量DataFrame
        "poa_direct": poa_direct,
        "poa_sky_diffuse": poa_sky_diffuse,
        "poa_ground_diffuse": poa_ground_diffuse,
        "poa_total": poa_total
    }, index=meteo_df.index)

    total_baseline_kwh = np.nansum(p_baseline) / 1000  # 无雪基准年发电量（kWh）
    total_actual_kwh = np.nansum(p_actual) / 1000  # 考虑积雪的实际年发电量（kWh）

    return total_actual_kwh, total_baseline_kwh, poa_components, p_actual, p_baseline  # 返回实际发电量、基准发电量、辐照分量、实际功率、基准功率


def analyze_irradiance_components(poa_components, station):  # 分析倾斜面辐照分量的年度和月度统计
    day_mask = poa_components["poa_total"] > 1  # 仅统计白天有效时段（POA > 1 W/m2）
    poa_day = poa_components[day_mask]  # 筛选白天数据

    annual_total = poa_day.sum() / 1000  # 年度总量（kWh/m2）
    annual_ratio = annual_total / annual_total["poa_total"] * 100  # 各分量占比（%）

    print(f"\n=== {station} 全年辐照分量统计（仅白天有效时段） ===")
    print(f"总倾斜面辐照量: {annual_total['poa_total']:.2f} kWh/m²")
    print(f"直射分量: {annual_total['poa_direct']:.2f} kWh/m²  占比: {annual_ratio['poa_direct']:.1f}%")
    print(f"天空散射分量: {annual_total['poa_sky_diffuse']:.2f} kWh/m²  占比: {annual_ratio['poa_sky_diffuse']:.1f}%")
    print(f"雪面反射分量: {annual_total['poa_ground_diffuse']:.2f} kWh/m²  占比: {annual_ratio['poa_ground_diffuse']:.1f}%")

    monthly_total = poa_day.resample('M').sum() / 1000  # 按月重采样并求和
    monthly_ratio = monthly_total.div(monthly_total['poa_total'], axis=0) * 100  # 计算占比
    monthly_ratio.index = monthly_ratio.index.month  # 用月份数字作为索引
    monthly_ratio = monthly_ratio.fillna(0)  # 填充缺失值

    print(f"\n=== {station} 分月辐照分量占比 (%)（仅白天有效时段） ===")
    print(monthly_ratio.round(1).to_string())

    return annual_total, monthly_ratio  # 返回年度总量和月度占比


def find_best_angles(station, station_data, year=2023, azimuth_range=None, tilt_range=None):  # 参数：站点参数字典、模拟年份
    meteo_df = simulate_antarctic_meteorology(station_data, year)  # 生成气象数据（只需生成一次，所有角度共用）
    results = []  # 存储所有角度组合的结果
    print(f"\n正在计算{station}角度组合（共{len(azimuth_range)*len(tilt_range)}组）...")
    for azimuth in azimuth_range:  # 遍历所有方位角
        for tilt in tilt_range:  # 遍历所有倾角
            actual_kwh, baseline_kwh, _, _, _ = calculate_marion_model_power(  # 计算该角度组合的发电量
                station_data, meteo_df, tilt, azimuth  # 参数：站点参数、气象数据、组件倾角、组件方位角（正北为0）
            )
            results.append({
                'azimuth': azimuth,
                'tilt': tilt,
                'annual_generation_kwh': actual_kwh,
                'baseline_generation_kwh': baseline_kwh
            })
    return pd.DataFrame(results), meteo_df  # 返回结果DataFrame和气象数据


def visualize_results(results_df, station, year, meteo_df, best_azimuth, best_tilt, monthly_ratio):  # 可视化分析结果，生成6个子图的综合分析图表
    gen_matrix = results_df.pivot(  # 将结果数据透视成矩阵形式（倾角x方位角）
        index='tilt',
        columns='azimuth',
        values='annual_generation_kwh'
    )
    colors = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']  # 自定义颜色映射（浅绿到深蓝）
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)  # 创建自定义颜色映射

    fig = plt.figure(figsize=(24, 20))  # 创建大图（3行3列，实际使用6个子图）

    ax1 = plt.subplot(3, 3, 1)  # 子图1：年发电量热力图
    sns.heatmap(
        gen_matrix,
        cmap=cmap,
        annot=True,  # 显示数值标注
        fmt=".1f",  # 保留1位小数
        cbar_kws={'label': '年发电量 (kWh)'},
        vmin=results_df['annual_generation_kwh'].min(),
        vmax=results_df['annual_generation_kwh'].max(),
        ax=ax1
    )
    max_gen = results_df.loc[results_df['annual_generation_kwh'].idxmax(), 'annual_generation_kwh']  # 获取最大发电量
    ax1.scatter(  # 标记最佳角度位置
        best_azimuth,
        best_tilt,
        color='red',
        s=400,
        marker='*',
        label=f'最佳角度\n方位角: {best_azimuth:.1f}°\n倾角: {best_tilt:.1f}°\n年发电量: {max_gen:.2f} kWh'
    )
    ax1.set_title(f'{year}年 南极{station}光伏年发电量热力图（最终校准版）', fontsize=12, fontweight='bold')  # 设置标题
    ax1.set_xlabel('方位角 (°，正北为0°)')  # 设置X轴标签
    ax1.set_ylabel('倾角 (°)')  # 设置Y轴标签
    ax1.legend(loc='upper right')  # 显示图例

    ax2 = plt.subplot(3, 3, 2)  # 子图2：月均积雪深度
    monthly_snow = meteo_df['snow_depth_cm'].resample('M').mean()  # 按月平均
    ax2.bar(range(1, 13), monthly_snow.values, color='steelblue', alpha=0.7)  # 绘制柱状图
    ax2.set_title(f'{station}全年月均地面积雪深度', fontsize=12, fontweight='bold')
    ax2.set_xlabel('月份')
    ax2.set_ylabel('积雪深度 (cm)')
    ax2.set_xticks(range(1, 13))
    ax2.grid(alpha=0.3, axis='y')

    ax3 = plt.subplot(3, 3, 3)  # 子图3：雪损率热力图
    results_df['snow_loss_rate'] = (results_df['baseline_generation_kwh'] - results_df['annual_generation_kwh']) / results_df['baseline_generation_kwh']  # 计算雪损率 = (基准发电量 - 实际发电量) / 基准发电量
    loss_matrix = results_df.pivot(index='tilt', columns='azimuth', values='snow_loss_rate')  # 透视成矩阵
    sns.heatmap(
        loss_matrix,
        cmap='Reds',  # 红色系表示损失
        annot=True,  # 显示数值标注
        fmt=".2f",
        cbar_kws={'label': '雪损率（0-1）'},
        ax=ax3
    )
    ax3.set_title(f'{station}各角度雪损率分布', fontsize=12, fontweight='bold')
    ax3.set_xlabel('方位角 (°，正北为0°)')
    ax3.set_ylabel('倾角 (°)')

    ax4 = plt.subplot(3, 3, 4)  # 子图4：月均雪密度
    monthly_density = meteo_df['snow_density_gcm3'].resample('M').mean()  # 按月平均
    ax4.plot(range(1, 13), monthly_density.values, color='darkorange', linewidth=2, marker='o')  # 绘制折线图
    ax4.set_title(f'{station}全年月均雪密度', fontsize=12, fontweight='bold')
    ax4.set_xlabel('月份')
    ax4.set_ylabel('雪密度 (g/cm³)')
    ax4.set_xticks(range(1, 13))
    ax4.grid(alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)  # 子图5：雪状态月度分布
    monthly_state = meteo_df.groupby([meteo_df.index.month, 'snow_state']).size().unstack(fill_value=0)  # 按月统计各雪状态的小时数
    monthly_state_pct = monthly_state.div(monthly_state.sum(axis=1), axis=0) * 100  # 转换为百分比
    monthly_state_pct.plot(kind='bar', stacked=True, ax=ax5, colormap='Set2')  # 绘制堆叠柱状图
    ax5.set_title(f'{station}雪状态月度分布（小时占比）', fontsize=12, fontweight='bold')
    ax5.set_xlabel('月份')
    ax5.set_ylabel('占比 (%)')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.set_xticks(range(12))
    ax5.set_xticklabels(range(1, 13))

    ax6 = plt.subplot(3, 3, 6)  # 子图6：分月辐照分量占比
    monthly_ratio[['poa_direct', 'poa_sky_diffuse', 'poa_ground_diffuse']].plot(  # 绘制堆叠柱状图
        kind='bar',
        stacked=True,
        ax=ax6,
        color=['#ff7f0e', '#2ca02c', '#1f77b4'],
        alpha=0.8
    )
    ax6.set_title(f'{station}分月辐照分量占比（仅白天）', fontsize=12, fontweight='bold')
    ax6.set_xlabel('月份')
    ax6.set_ylabel('占比 (%)')
    ax6.legend(['直射', '天空散射', '雪面反射'], bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.set_xticks(range(12))
    ax6.set_xticklabels(range(1, 13))
    ax6.set_ylim(0, 105)  # 设置Y轴范围

    plt.tight_layout()  # 调整布局
    plt.savefig(f'antarctic_marion_model_{station}_{year}_best_final.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()  # 显示图表


if __name__ == "__main__":  # 主程序入口
    print("=== 南极科考站光伏最佳角度计算器（最终收敛版·尽量贴近真实规律）===")  # 程序标题
    print("=== 修正：极夜发电、辐照占比、积雪峰值、雪密度、冰壳占比、倾角饱和、发电量量级 ===")

    print("\n请选择科考站：")  # 选择科考站
    print("1. 长城站（乔治王岛，季节性积雪区）")  # 长城站选项
    print("2. 中山站（南极大陆，常年积雪区）")  # 中山站选项
    while True:
        choice = input("请输入选项(1/2)：")  # 获取用户选择
        if choice in ['1', '2']:
            station = "长城站" if choice == '1' else "中山站"  # 根据选择确定站点
            station_data = STATIONS[station]  # 获取站点参数
            break
        else:
            print("输入错误，请选择1或2！")

    print(f"\n=== 所选站点基础信息 ===")  # 显示站点基础信息
    print(f"站点名称: {station}")  # 站点名称
    print(f"地理坐标: {abs(station_data['latitude'])}°S, {abs(station_data['longitude'])}°{'W' if station=='长城站' else 'E'}")  # 地理坐标
    print(f"年平均气温: {station_data['temperature']}℃")  # 年平均气温
    print(f"光伏板额定功率: {station_data['P_DC0']/1000}kW")  # 光伏板额定功率
    print(f"安装参数: 组件高度{station_data['MODULE_HEIGHT']}cm | 底部离地{station_data['MODULE_CLEARANCE']}cm")  # 安装参数
    print(f"积雪反照率: {station_data['snow_albedo']}（纯积雪）")  # 积雪反照率

    year = validate_input("\n请输入计算年份", 2000, 2100, int, default=2023)  # 输入计算年份
    pv_power = validate_input(  # 输入光伏板额定功率
        "请输入光伏板额定功率 (kW)（单块）", 0.1, 10.0, float, default=station_data["P_DC0"]/1000
    )
    station_data["P_DC0"] = pv_power * 1000  # 转换为W

    print("\n=== 角度搜索范围设置 ===")  # 设置角度搜索范围
    print("提示：南半球光伏最佳方位角为0°（正北），建议搜索范围0-360°")
    start_azimuth = validate_input("方位角起始值 (0-360)", 0, 360, float, default=0)  # 方位角起始值
    end_azimuth = validate_input("方位角结束值 (0-360)", start_azimuth, 360, float, default=360)  # 方位角结束值
    step_azimuth = validate_input("方位角步长 (0.1-90)", 0.1, 90, float, default=30)  # 方位角步长

    print("\n提示：高纬度地区最佳倾角通常为50-70°，建议搜索范围0-90°")
    start_tilt = validate_input("倾角起始值 (0-90)", 0, 90, float, default=0)  # 倾角起始值
    end_tilt = validate_input("倾角结束值 (0-90)", start_tilt, 90, float, default=90)  # 倾角结束值
    step_tilt = validate_input("倾角步长 (0.1-30)", 0.1, 30, float, default=10)  # 倾角步长

    azimuth_num = int(round((end_azimuth - start_azimuth) / step_azimuth)) + 1  # 计算方位角点数
    azimuth_range = np.linspace(start_azimuth, end_azimuth, azimuth_num, endpoint=True)  # 生成方位角搜索网格
    tilt_num = int(round((end_tilt - start_tilt) / step_tilt)) + 1  # 计算倾角点数
    tilt_range = np.linspace(start_tilt, end_tilt, tilt_num, endpoint=True)  # 生成倾角搜索网格

    results_df, meteo_df = find_best_angles(station, station_data, year, azimuth_range, tilt_range)  # 执行角度搜索

    max_gen_idx = results_df['annual_generation_kwh'].idxmax()  # 确定最大发电量索引
    best_azimuth = results_df.loc[max_gen_idx, 'azimuth']  # 最佳方位角
    best_tilt = results_df.loc[max_gen_idx, 'tilt']  # 最佳倾角
    _, _, poa_components, _, _ = calculate_marion_model_power(  # 使用最佳角度重新计算，获取详细结果
        station_data, meteo_df, best_tilt, best_azimuth
    )
    annual_total, monthly_ratio = analyze_irradiance_components(poa_components, station)  # 参数：辐照分量DataFrame、站点名称

    max_generation = results_df.loc[max_gen_idx, 'annual_generation_kwh']  # 最大发电量
    baseline_generation = results_df.loc[max_gen_idx, 'baseline_generation_kwh']  # 基准发电量
    snow_loss_rate = (baseline_generation - max_generation) / baseline_generation * 100  # 相对雪损率（%）

    visualize_results(results_df, station, year, meteo_df, best_azimuth, best_tilt, monthly_ratio)  # 参数：结果数据、站点、年份、气象数据、最佳角度、月度占比

    print("\n" + "="*80)  # 输出最终结果
    print(f" 最终收敛版 {station}最佳角度计算结果（{year}年）")  # 结果标题
    print("="*80)
    print(f" 光伏板参数：额定功率 {pv_power}kW | 温度系数 {station_data['GAMMA']}/℃")  # 光伏板参数
    print(f" 最佳方位角：{best_azimuth:.1f}°（正北为0°）")
    print(f" 最佳倾角：{best_tilt:.1f}°")
    print(f" 无雪基准年发电量：{baseline_generation:.2f} kWh")  # 无雪基准年发电量
    print(f" 考虑积雪实际年发电量：{max_generation:.2f} kWh")  # 考虑积雪实际年发电量
    print(f" 累计积雪损失电量：{baseline_generation - max_generation:.2f} kWh")  # 累计积雪损失电量
    print(f" 相对雪损率：{snow_loss_rate:.2f}%")  # 相对雪损率
    print("="*80)
    print(f" 结果图表已保存：antarctic_marion_model_{station}_{year}_best_final.png")  # 结果图表保存路径