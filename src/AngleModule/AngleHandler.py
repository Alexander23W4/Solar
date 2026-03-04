import pandas as pd
import pvlib
import numpy as np

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

            # function：arccos(cos(elevation) * cos(azimuth - angle))
            cos_value = np.cos(elev_rad) * np.cos(az_diff_rad)

            # ensure that cos_value is  between [-1, 1]
            cos_value = np.clip(cos_value, -1.0, 1.0)

            # cal arccos
            result_rad = np.arccos(cos_value)
            result_deg = np.degrees(result_rad)
            return result_deg
        else:
            # sun is below the horizon , return -1000
            return -1000.0




if __name__ == "__main__":
    handler = AngleHandler()
    result = handler.getAngle('2025-06-22 9:00')
    print(result)