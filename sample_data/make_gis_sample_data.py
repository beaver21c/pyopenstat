"""
Generate realistic sample GIS dataset for Korea (Seoul 25 Autonomous Districts)
Variables: region_id, region_name, latitude, longitude, welfare_expenditure, elderly_ratio, poverty_rate, single_household, welfare_centers
"""

import pandas as pd
import numpy as np

districts = [
    ("종로구", 37.5730, 126.9794, 98.5, 18.2, 3.8, 38.2, 5),
    ("중구", 37.5641, 126.9979, 102.1, 19.5, 4.1, 41.5, 4),
    ("용산구", 37.5326, 126.9906, 92.4, 17.1, 3.2, 38.0, 5),
    ("성동구", 37.5634, 127.0368, 88.6, 16.2, 2.9, 34.5, 6),
    ("광진구", 37.5385, 127.0822, 85.2, 15.6, 2.8, 37.8, 6),
    ("동대문구", 37.5744, 127.0396, 96.8, 18.5, 4.2, 39.4, 7),
    ("중랑구", 37.6063, 127.0926, 108.4, 20.1, 5.8, 35.1, 8),
    ("성북구", 37.5894, 127.0167, 91.2, 17.8, 3.5, 34.0, 7),
    ("강북구", 37.6398, 127.0255, 115.6, 22.3, 6.5, 35.8, 7),
    ("도봉구", 37.6688, 127.0471, 104.2, 21.0, 4.9, 31.2, 6),
    ("노원구", 37.6542, 127.0568, 112.0, 18.9, 5.5, 29.5, 9),
    ("은평구", 37.6027, 126.9291, 101.5, 19.4, 4.6, 32.8, 8),
    ("서대문구", 37.5791, 126.9368, 89.4, 17.5, 3.3, 37.2, 6),
    ("마포구", 37.5663, 126.9016, 82.5, 15.1, 2.6, 38.9, 6),
    ("양천구", 37.5170, 126.8665, 87.2, 15.3, 3.4, 26.5, 7),
    ("강서구", 37.5509, 126.8495, 106.8, 17.6, 5.9, 36.1, 10),
    ("구로구", 37.4954, 126.8874, 94.2, 17.9, 3.9, 31.8, 7),
    ("금천구", 37.4568, 126.8955, 101.2, 18.8, 4.8, 38.5, 6),
    ("영등포구", 37.5264, 126.8962, 86.8, 16.4, 3.1, 37.0, 6),
    ("동작구", 37.5124, 126.9393, 84.5, 16.8, 3.0, 37.9, 6),
    ("관악구", 37.4784, 126.9516, 95.6, 17.2, 4.3, 51.2, 7),
    ("서초구", 37.4836, 127.0327, 72.4, 14.1, 1.8, 29.8, 5),
    ("강남구", 37.5172, 127.0473, 76.5, 14.5, 2.1, 32.5, 6),
    ("송파구", 37.5145, 127.1058, 81.2, 14.8, 2.4, 30.1, 7),
    ("강동구", 37.5301, 127.1238, 88.9, 16.5, 3.3, 31.0, 6),
]

data = []
for idx, (name, lat, lon, exp, eld, pov, sng, cnt) in enumerate(districts, start=1):
    data.append({
        "region_id": idx,
        "region_name": name,
        "latitude": lat,
        "longitude": lon,
        "welfare_expenditure": exp,
        "elderly_ratio": eld,
        "poverty_rate": pov,
        "single_household": sng,
        "welfare_centers": cnt
    })

df = pd.DataFrame(data)
df.to_csv("/mnt/share/temp/pyjamovi/sample_data/Korea_Regional_Welfare_GIS.csv", index=False)
print("Created sample_data/Korea_Regional_Welfare_GIS.csv with", len(df), "rows.")
