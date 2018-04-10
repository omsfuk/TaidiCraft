# -*- encoding: utf-8 -*-

import json

with open("emperor.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

provinces = []
cities = []
areas = []
for province in obj:
    provinces.append(province['name'])
    for city in province['city']:
        cities.append(city['name'])
        for area in city['area']:
            areas.append(area)

"""
with open("provinces") as f:
    json.dump(province, f, ensure_ascii=False, indent=4)
"""

"""
with open("cities.json", "w") as f:
    json.dump(cities, f, ensure_ascii=False, indent=4)
"""
with open("areas.json", "w") as f:
    json.dump(areas, f, ensure_ascii=False, indent=4)
