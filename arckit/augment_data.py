import json
import numpy as np

# Open and read the JSON file
with open('arc1.json', 'r') as file:
    data = json.load(file)

# Print the data
#print(data["train"]["780d0b14"]["train"][0]["input"])

inputarray = np.array(data["train"]["780d0b14"]["train"][0]["input"])

outputarray = np.array(data["train"]["780d0b14"]["test"][0]["output"])

print(inputarray)

#transformed = json.dumps(inputarray)

print(outputarray)


#with open("transformations.json", "w") as outfile:
    #json.dump(transformed, outfile)

with open("transformations.json", "w") as outfile:
    json.dump("hello", outfile)