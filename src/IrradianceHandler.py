import pvlib
import pandas as pd
import numpy as np

class SolarIrradiance:
    def __init__(self, latitude=-69.367, longitude=76.367, tz='Asia/Shanghai', linke_turbidity=3):
        self.latitude = latitude
        self.longitude = longitude
        self.tz = tz
        self.linke_turbidity = linke_turbidity
        self.location = pvlib.location.Location(latitude=self.latitude,
                                                longitude=self.longitude,
                                                tz=self.tz)

    def get_times(self, date_str='2025-12-22 12:00'):
        return pd.date_range(date_str, periods = 1, tz=self.tz)

    def get_clearsky(self, date_str='2025-12-22 12:00'):
        times = self.get_times(date_str)
        return self.location.get_clearsky(times, model = 'ineichen', linke_turbidity = self.linke_turbidity)

    def get_dni(self, date_str='2025-12-22 12:00'):
        clearsky = self.get_clearsky(date_str)
        return clearsky['dni'].iloc[0]  # 返回单个数值 dni

    def get_ghi(self, date_str='2025-12-22 12:00'):
        clearsky = self.get_clearsky(date_str)
        return clearsky['ghi'].iloc[0]

    def get_dhi(self, date_str='2025-12-22 12:00'):
        clearsky = self.get_clearsky(date_str)
        return clearsky['dhi'].iloc[0]

    def hay_davies_diffuse(self, DHI, DNI, GHI, solar_zenith_deg, aoi_deg, tilt_deg=90, albedo=0.2):
        z = np.radians(solar_zenith_deg)
        aoi = np.radians(aoi_deg)
        beta = np.radians(tilt_deg)

        # 各向异性指数
        denom = (GHI - DHI)
        if denom <= 0:
            A = 0.0
        else:
            A = (DNI * np.cos(z)) / denom

        # 天空散射部分
        sky_diffuse = DHI * (
                (1 - A) * (1 + np.cos(beta)) / 2
                + A * (np.cos(aoi) / np.cos(z))
        )

        # 地面反射部分
        ground_diffuse = GHI * albedo * (1 - np.cos(beta)) / 2

        # 总散射 = 天空散射 + 地面反射
        Idiff_tilt = sky_diffuse + ground_diffuse
        """
        计算倾斜面上的散射辐照度 (Hay & Davies模型) + 地面反射

        parameters
            DHI : float, 水平散射 (W/m2)
            DNI : float, 直接法向 (W/m2)
            GHI : float, 水平总辐照度 (W/m2)
            solar_zenith_deg : float, 太阳天顶角 (deg)
            aoi_deg : float, 面板与太阳的入射角 (deg)
            tilt_deg : float, 面板倾角 (deg), 默认=90 (垂直)
            albedo : float, 地面反照率, 默认0.2

        return
            Idiff_tilt : float, 倾斜面散射辐照度 (W/m2)
        """
        return Idiff_tilt


if __name__ == "__main__":
    solar = SolarIrradiance()
    print("DNI:", solar.get_dni())
    print("GHI:", solar.get_ghi())
    print("DHI:", solar.get_dhi())
