import json
import numpy as np

def rotate(array):
    rotated_90 = np.rot90(array, k=1).tolist()
    return rotated_90


# Open and read the JSON file
with open('arc1.json', 'r') as file:
    data = json.load(file)
    print(type(data))


with open("augmented_data.json", "w") as outfile:
    outer={}


    for id in data["train"].keys():
        aug_dict = {"train": [], "test": []}
        for training_item in range(len(data["train"][id]["train"])):
            input_array = np.array(data["train"][id]["train"][training_item]["input"])
            output_array = np.array(data["train"][id]["train"][training_item]["output"])
            aug_dict["train"].append({"input": rotate(input_array), "output": rotate(output_array)})
        for testing_item in range(len(data["train"][id]["test"])):
            input_array = np.array(data["train"][id]["test"][testing_item]["input"])
            output_array = np.array(data["train"][id]["test"][testing_item]["output"])
            aug_dict["test"].append({"input": rotate(input_array), "output": rotate(output_array)})
        outer[id]=aug_dict
 
    json.dump(outer, outfile)


'''
for id in data["train"].keys():
    # print(f"id: {id}")
    aug_dict = {}
    for training_item in data["train"][id]["train"]:
        input_array = np.array(data["train"][id]["train"][0]["input"])
        print(input_array)
        aug_dict[id] = {"train": [{"input": rotate(input_array)}]}
        output_array = np.array(data["train"][id]["train"][0]["output"])
        print(output_array)
            #augment_90(input_array, output_array)

    #output_array = np.array(data["train"][id]["test"])
'''

'''
print("\nORIGINAL:\n")
print(data["train"]["780d0b14"])

print("\n ROTATED 90:\n")
id = "780d0b14"
aug_dict = {id: {"train": [], "test": []}}
#aug_dict[id]["train"] = []
for training_item in range(len(data["train"][id]["train"])):
    input_array = np.array(data["train"][id]["train"][training_item]["input"])
    output_array = np.array(data["train"][id]["train"][training_item]["output"])
    aug_dict[id]["train"].append({"input": rotate(input_array), "output": rotate(output_array)})
for testing_item in range(len(data["train"][id]["test"])):
    input_array = np.array(data["train"][id]["test"][testing_item]["input"])
    output_array = np.array(data["train"][id]["test"][testing_item]["output"])
    aug_dict[id]["test"].append({"input": rotate(input_array), "output": rotate(output_array)})
print(aug_dict["780d0b14"])

#print(data["train"][id]["train"][0]["input"])

#print(data["train"][id]["train"])







# Print the data
#print(data["train"]["780d0b14"]["train"][0]["input"])

inputarray = np.array(data["train"]["780d0b14"]["train"][0]["input"])

outputarray = np.array(data["train"]["780d0b14"]["test"][0]["output"])

#print(inputarray)

#transformed = json.dumps(inputarray)

#print(outputarray)


#with open("transformations.json", "w") as outfile:
    #json.dump(transformed, outfile)

#with open("transformations.json", "w") as outfile:
#    json.dump("hello", outfile)
'''