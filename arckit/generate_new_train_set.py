

import json
import numpy as np
import os

def check_and_open_json(id):
    directory = "/Users/mette/dreamcoder-arc-1/arckit/re_arc/tasks"
    json_data=None
    for filename in os.listdir(directory):

        #split filename to sepearte the id
        file_id = os.path.splitext(filename)[0]
        if file_id == id:
            file_path = os.path.join(directory,filename)
            with open(file_path, "r") as right_json:
                json_data=json.load(right_json)
    
    return json_data
    


#Open new file named "new_train_set" if not already there in write mode
with open("new_train_set.json", "w") as new_file:

#Open Arc1.json, open augmented_data.json, open each re_arc/tasks i read mode
    with open("arc1.json", "r") as read_file:
        data=json.load(read_file)

        outer={}

        for i, id in enumerate(data["train"].keys()):
            print(i)
            
            #fetch the right json. Open json
            right_json=check_and_open_json(id) #pick and open right json
            if right_json==None: #if new sample not generate for id, skip.
                continue
      
            aug_dict = {"train": [], "test": []}
            
         
            for i, item in enumerate(range(len(right_json)-1)): #subtract one for to save it for test
                input_array = right_json[i]["input"]
                output_array = right_json[i]["output"]
                aug_dict["train"].append({"input": input_array, "output": output_array})
    
            last_element=right_json[-1]
            test_input_array = last_element["input"]
            test_output_array = last_element["output"]
            aug_dict["test"].append({"input": test_input_array, "output": test_output_array}) #
      
            outer[f"{id} {i}"]=aug_dict
            json.dump(outer, new_file)
     
        
    
