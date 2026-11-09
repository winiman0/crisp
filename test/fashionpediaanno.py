import json
data = json.load(open("dataset/fashionpedia/instances_attributes_val2020.json"))
print(list(data.keys()))
for c in data["categories"]:
    print(c["id"], c["name"], c.get("supercategory"))
print(len(data["images"]), len(data["annotations"]))