import pandas as pd
import pvlib
import numpy as np

from AssemblyModule.ComponentsHandler import AngleGenerator
# python3 -m AngleModule.AngleHandler   # 这个文件地下的测试要这样才能运行

class AngleHandler:   
    """
    AngleHandler:  calculate the included angle between the light and the board

    Attributes:
        latitude 
        longitude
        timeZone
        (These elements are provided for pvlib for locating)
    """
    def __init__(self, latitude=-62.12, longitude=-58.57, timeZone='Etc/GMT+3'):
        self.latitude = latitude
        self.longitude = longitude
        self.timeZone = timeZone

    def getAngle(self, moment):  
        """getAngle
        Attributes:
            moment  e.g. time='2025-06-22 9:00'

        Return: (two critical angles)
            apparent_elevation
            azimuth
        """
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
        Transfer input temporal elements to standard string
        Returns:
        str: standard string, e.g.'2025-06-22 11:00'
        """
        time_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
        return time_str

    def is_within_90(self, moment, angle):
        """
        judge whether the included angle between given angle and the azimuth(sun) less than 90 degree
        (if larger than 90 degree, the face can not get any direction irradiance)

        Returns:
        bool: True -> within ±90°, False -> beyond ±90°
        """
        solar_data = self.getAngle(moment)
        azimuth = solar_data['azimuth'].iloc[0]
        diff = (azimuth - angle + 180) % 360 - 180  

        return abs(diff) <= 90


    def AngleCombination(self, moment, angle):
        """
        Compute the combined angle value of the sun's position with a given angle setting.

        Parameters:
            moment: A time string in the format 'YYYY-MM-DD HH:MM', e.g., '2025-06-22 9:00'
            angle: An input reference angle in degrees

        Returns:
            float: The combined angle value in degrees; returns -1000 if the sun is below the horizon
        """
        # get solar position data
        solar_data = self.getAngle(moment)
        apparent_elevation = solar_data['apparent_elevation'].iloc[0]
        azimuth = solar_data['azimuth'].iloc[0]

        # judge whether the sun beyonds horizontal level
        if apparent_elevation > 0:
            # change angle -> radians
            elev_rad = np.radians(apparent_elevation)
            az_diff_rad = np.radians(azimuth - angle)

            # function：arccos(cos(elevation) * cos(azimuth - angle))   这个是计算组合角的公式 cos*cos
            cos_value = np.cos(elev_rad) * np.cos(az_diff_rad)

            # ensure that cos_value is  between [-1, 1]
            cos_value = np.clip(cos_value, -1.0, 1.0)

            # cal arccos
            result_rad = np.arccos(cos_value)      # 圆周角结果(rad为单位)
            result_deg = np.degrees(result_rad)    # 角度角结果(度为单位)
            return result_deg
        else:
            # sun is below the horizon , return -1000
            return -1000.0


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
# 这部分的原理就是这些: 得到组合角
# 注意: 这里pvlib库生成的 : azimuth方位角 -> 0度为正北,顺时针方向增加;  apparent_elevation俯仰角, 平行于地面为0度, 向上角度增加
#      AngleGenerator 在生成 face_angles的时候, base_angle是第一个面的朝向, 数值是以正北为0度, 顺时针偏移的方向.
# AngleHandler.getAngle (lat, lon, test_time -> apparent_elevation, azimuth) 
# AngleGenerator (face angles)
# AngleHandler.AngleCombination (face angles, test_time -> incidence_angles, Illuminated/In_Shadow)
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

# python3 -m AngleModule.AngleHandler   # 测试要这样才能运行
if __name__ == "__main__":
    # handler = AngleHandler()
    # result = handler.getAngle('2025-06-22 9:00')
    # print(result)


    # 1. Convert coordinates: 31°27'5" N, 119°29'0" E to decimal degrees
    lat = 31 + 27/60 + 5/3600
    lon = 119 + 29/60 + 0/3600
    test_time = '2026-04-26 08:30'
    
    # 2. Initialize AngleHandler with target location parameters
    handler = AngleHandler(latitude=lat, longitude=lon, timeZone='Asia/Shanghai')
    
    # 3. Initialize AngleGenerator for a 3-sided prism starting at 0 degrees   (From ComponentsHandler)
    prism_gen = AngleGenerator(n=3, base_angle=25)
    face_angles = prism_gen.generate()  # Expected: [0.0, 120.0, 240.0]
    
    # 4. Fetch and print the baseline solar data
    sol_pos = handler.getAngle(test_time)
    print(f"--- Location: Lat {lat:.4f}°, Lon {lon:.4f}° ---")
    print(f"--- Simulation Time: {test_time} ---")
    print(f"Solar Elevation: {sol_pos['apparent_elevation'].iloc[0]:.2f}°")
    print(f"Solar Azimuth:   {sol_pos['azimuth'].iloc[0]:.2f}°\n")
    
    print("--- Incidence Angle Results for Each Prism Face ---")
    # 5. Calculate and log metrics for each distinct prism orientation
    for i, angle in enumerate(face_angles):
        # Evaluate incidence angle using your original formula
        incidence_angle = handler.AngleCombination(test_time, angle)
        
        # Verify sun visibility using your original boundary threshold logic
        is_lit = handler.is_within_90(test_time, angle)
        status = "(Illuminated)" if is_lit else "(In Shadow)"
        
        print(f"Face {i+1} (Heading {angle:5.1f}°): Incidence Angle = {incidence_angle:6.2f}° {status}")
            