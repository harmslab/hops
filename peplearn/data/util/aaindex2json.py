# hacked script for converting aaindex1 file to json

"""
H ANDN920101
D alpha-CH chemical shifts (Andersen et al., 1992)
R PMID:1575719
A Andersen, N.H., Cao, B. and Chen, C.
T Peptide/protein structure analysis using the chemical shift index method: 
  upfield alpha-CH values reveal dynamic helices and aL sites
J Biochem. and Biophys. Res. Comm. 184, 1008-1014 (1992)
C BUNA790102    0.949
I    A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
    4.35    4.38    4.75    4.76    4.65    4.37    4.29    3.97    4.63    3.95
    4.17    4.36    4.52    4.66    4.44    4.50    4.35    4.70    4.60    3.95
"""


import sys, copy, re, json

template_dict = {"H":[],
                 "D":[],
                 "R":[],
                 "A":[],
                 "*":[],
                 "T":[],
                 "J":[],
                 "C":[],
                 "I":[]}

f = open(sys.argv[1],'r')
lines = f.readlines()
f.close()

clean_up_pattern = re.compile("\"")

out_dict = {}
current_dict = copy.deepcopy(template_dict)
for l in lines:

    if l.startswith("//"):


        # Deal with meta data
        name = " ".join(current_dict["H"])
        name = clean_up_pattern.sub("'",name)

        description = " ".join(current_dict["D"])
        description = clean_up_pattern.sub("'",description)

        citation = "{} '{}' {}".format(" ".join(current_dict["A"]),
                                       " ".join(current_dict["T"]),
                                       " ".join(current_dict["J"]))

        citation = citation + "; Kawashima, S. and Kanehisa, M. 'AAindex: amino acid index database.'  Nucleic Acids Res. 28, 374 (2000)."

        citation = clean_up_pattern.sub("'",citation)

        notes = " ".join(current_dict["*"])
        notes = clean_up_pattern.sub("'",notes)
        
        # parse amino acid data
        aa_lines = current_dict["I"]

        aa_names = aa_lines[0].split()
        row_0_names = [aa.split("/")[0] for aa in aa_names]
        row_1_names = [aa.split("/")[1] for aa in aa_names]

        row_0_values = aa_lines[1].split()
        row_1_values = aa_lines[2].split()

        values = {}
        for i in range(len(row_0_values)):
            try:
                values[row_0_names[i]] = float(row_0_values[i])
            except ValueError:
                values[row_0_names[i]] = "NA"

            try:
                values[row_1_names[i]] = float(row_1_values[i])
            except ValueError:
                values[row_1_names[i]] = "NA"

        # look for duplicate name entries
        try:
            out_dict[name]
            err = "duplicate value name ({})".format(name)
            raise ValueError
        except KeyError:
            pass

        out_dict[name] = {"description":description,
                          "refs":citation,
                          "notes":notes,
                          "values":values}

        current_dict = copy.deepcopy(template_dict)
        continue

    this_entry = l[0]
    if l[0] != " ":
        current_entry = this_entry

    current_dict[current_entry].append(l[1:].strip())


print(json.dumps(out_dict,indent=4,sort_keys=True))
    
    
