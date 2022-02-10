
import yaml
import json
from collections import defaultdict
with open("test.yaml", 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
print(data)
result =defaultdict(dict)
for i in range(len(data)):
    result[str(i+1)] = data[i+1]
jstring = json.dumps(result, indent=4)
with open("test.json", "w") as f:
    f.write(jstring)

