import pandas as pd
import pvlib
import numpy as np
class AngleHandler:   # input latitude longitude and timeZone to pin your position
    def __init__(self, latitude=-62.12, longitude=-58.57, timeZone='Etc/GMT+3'):
        self.latitude = latitude
        self.longitude = longitude
        self.timeZone = timeZone

        #function "getAngle", input moment, return apparent_elevation(angle) & azimuth angle
    def getAngle(self, moment):  # e.g. time='2025-06-22 9:00'
        location = pvlib.location.Location(latitude=self.latitude,
                                           longitude=self.longitude,
                                           tz=self.timeZone)
        times = pd.date_range(moment, periods=1, tz=self.timeZone)
        solpos = location.get_solarposition(times)
        return solpos[['apparent_elevation', 'azimuth']]

    def getApparent_elevation(self, moment):
        solar_data = self.getAngle(moment)
        return solar_data['apparent_elevation'].iloc[0]

    def getAzimuth(self, moment):
        solar_data = self.getAngle(moment)
        return solar_data['azimuth'].iloc[0]

    def timeTransfer(self, year, month, day, hour, minute):
        """
        将输入的年、月、日、时、分转换为标准时间字符串格式
        Returns:
        str: 格式化的时间字符串，如 '2025-06-22 11:00'
        """
        time_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
        return time_str

    def is_within_90(self, moment, angle):
        """
        判断给定角度与太阳方位角差值是否在 ±90° 内

        Returns:
        bool: True 表示差值在 ±90°，False 表示超过 ±90°
        """
        solar_data = self.getAngle(moment)
        azimuth = solar_data['azimuth'].iloc[0]
        diff = (azimuth - angle + 180) % 360 - 180  # 范围 [-180, 180]

        # 判断是否在 ±90°
        return abs(diff) <= 90


    def AngleCombination(self, moment, angle):
        # 获取太阳位置数据
        solar_data = self.getAngle(moment)
        apparent_elevation = solar_data['apparent_elevation'].iloc[0]
        azimuth = solar_data['azimuth'].iloc[0]

        # 判断太阳是否在地平线以上
        if apparent_elevation > 0:
            # 将角度转换为弧度进行计算
            elev_rad = np.radians(apparent_elevation)
            az_diff_rad = np.radians(azimuth - angle)

            # 计算公式：arccos(cos(elevation) * cos(azimuth - angle))
            cos_value = np.cos(elev_rad) * np.cos(az_diff_rad)

            # 确保cos_value在[-1, 1]范围内，避免数值误差
            cos_value = np.clip(cos_value, -1.0, 1.0)

            # 计算反余弦并转换为角度
            result_rad = np.arccos(cos_value)
            result_deg = np.degrees(result_rad)
            """
            计算太阳位置与给定角度组合后的角度值

            Parameters:
            moment: 时间字符串，如 '2025-06-22 9:00'
            angle: 给定的参考角度（度）

            Returns:
            float: 组合后的角度值（度），如果太阳在地平线以下则返回-1000
            """
            return result_deg
        else:
            # 太阳在地平线以下，返回-1000
            return -1000.0




if __name__ == "__main__":
    handler = AngleHandler()
    result = handler.getAngle('2025-06-22 9:00')
    print(result)