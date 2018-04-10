# -*- encoding: utf-8 -*-
import re
import json

p = re.compile(r"^[0-9].+$")

with open("emperor.txt", "r") as f:
    text = f.readlines()

ans = []
for txt in text:
    if not p.match(txt):
        ans = ans + txt.strip("\n").strip(" ").split(" ")

with open("emperor.json", "w") as f:
    json.dump(ans, f, ensure_ascii=False, indent=4)
