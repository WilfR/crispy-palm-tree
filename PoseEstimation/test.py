
import json


def jsonFileToDict(jsonFilename):
    with open(jsonFilename) as f:
        data = json.load(f)
    return data


jsonFile = 'f:\\home\\Data\\COCO\\annotations\\captions_val2017.json'
data = jsonFileToDict(jsonFile)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(data)
