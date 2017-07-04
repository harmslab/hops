# hacked script for converting my original csv format data to json 

import sys, copy, json

f = open(sys.argv[1],'r')
lines = f.readlines()
f.close()

data_types = lines[0].split()

aa_dict = {aa:0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}

data = {k:{"values":copy.deepcopy(aa_dict)} for k in data_types}

for l in lines[1:]:
    if l.startswith("Nterm"):
        break
    
    col = l.split()
    aa = col[0]
    for i in range(len(col[1:])):
        key = data_types[i]
        value = float(col[i+1])
        data[key]["values"][aa] = value

for d in data_types:
    data[d]["description"] = d
    data[d]["refs"] = ""
    data[d]["notes"] = ""
 
print(json.dumps(data,sort_keys=True,indent=4))
