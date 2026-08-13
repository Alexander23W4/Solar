from AngleModule.AngleHandler import AngleHandler
from AssemblyModule.ComponentsHandler import AngleGenerator
from Shade.ShadeHandler import CalShade

def get_prism_shadow_report(
    test_time, lat, lon, timezone, prisms_data, base_angle=0
):
    """
    Calculate and output the shadow area and ratio report, including AOI per face.
    """
    # 1. Initialize core handlers
    handler = AngleHandler(latitude=lat, longitude=lon, timeZone=timezone)
    shading_engine = CalShade()
    
    # 2. Retrieve solar position
    sol_pos = handler.getAngle(test_time)
    elevation = sol_pos['apparent_elevation'].iloc[0]
    azimuth = sol_pos['azimuth'].iloc[0]
    
    # 3. Calculate Incidence Angles (AOI) per face
    all_incidence_angles = []
    prism_gen = AngleGenerator(n=3, base_angle=base_angle)
    face_angles = prism_gen.generate()
    
    # Calculate AOI for each face across all prisms
    # We maintain the same order as the shading engine's face iteration
    for _ in prisms_data:
        for angle in face_angles:
            aoi = handler.AngleCombination(test_time, angle)
            all_incidence_angles.append(aoi)
        
    # 4. Compute shadows
    shadow_results = shading_engine.compute_single_time_shadows(
        prisms_params=prisms_data,
        solar_elevation=elevation,
        solar_azimuth=azimuth,
        incidence_angles=all_incidence_angles
    )
    
    # 5. Inject AOI into results for reporting
    for i, res in enumerate(shadow_results):
        res['AOI'] = all_incidence_angles[i]
        
    return shadow_results, elevation, azimuth

# --- Example Usage ---
if __name__ == "__main__":

    lat, lon = 31.45, 119.48    
    time_str = '2026-04-26 08:30'
    baseAngle = 25

    my_prisms = [
        {'center_x': 0.0, 'center_y': 0.0, 'width': 2.0, 'side_height': 3.0, 'rotation': baseAngle},
        {'center_x': 2.5, 'center_y': 0.0, 'width': 2.0, 'side_height': 3.0, 'rotation': baseAngle}
    ]

    
    results, elev, az = get_prism_shadow_report(
        test_time=time_str, lat=lat, lon=lon, 
        timezone='Asia/Shanghai', prisms_data=my_prisms, base_angle=baseAngle
    )
    
    # Print header
    print(f"Time: {time_str} | Solar Elevation: {elev:.2f}° | Solar Azimuth: {az:.2f}°")
    print(f"{'Prism':<8} | {'Face':<6} | {'AOI(°)':<8} | {'Area(m2)':<10} | {'Ratio(%)':<10} | {'Status'}")
    print("-" * 80)
    
    # Print rows
    for res in results:
        print(f"{res['Prism']:<8} | {res['Face']:<6} | {res['AOI']:<8.2f} | "
              f"{res['Shadow Area (m2)']:<10.4f} | {res['Shadow Ratio (%)']:<10.2f} | {res['Status']}")

        