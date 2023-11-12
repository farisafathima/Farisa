import pandas as pd
import numpy as np
import CSVProcessor
import JSONProcessor
import re

#QUESTION 1

df = CSVProcessor.load_csv('titanic.csv')
total_cols = CSVProcessor.num_cols((df))
print(f"The total num of cols : {total_cols}")
total_rows = CSVProcessor.num_rows(df)
print(f"The total num of rows: {total_rows}")
fill = CSVProcessor.fill_cols(df)
print(f"\n Missing values: {fill}")

#QUESTION 2

df2=pd.read_json('players.json')
print(df2)

max_player_row = df2.loc[df2["player_score"].idxmax()]
df2["man_of_match"] = df2.apply(lambda row: True if row["player"] == max_player_row["player"] else False, axis=1)
#axis=1 is cols
new_file = 'new_players.json'
df2.to_json(new_file,orient='records', lines=True)
print(df2)


#QUESTION 4
dna = "ATCGCGAATTCAC"
pattern = "GAATTC"
if re.search(pattern,dna):
    print("found a mmatch")
else:
    print("no match")

#QUESTION 5
dna = "ATCGCGAATTCAC"
pattern1 = "GGACC"
pattern2="GGTCC"
if re.search(pattern1 or pattern2 , dna):
    print("found a mmatch")
else:
    print("no match")

#QUESTION 6
dna = "ATCGCGAATTCAC"
pattern = "GC[A,T,G,C]GC"

if re.search(pattern, dna):
    print("Found a match")
else:
    print("No match")

#QUESTION 7
# ^:  the start of the string.
# AUG: Matches the start codon "AUG
# [AUGC]{30,1000}: Matches any combination of the nucleotides A, U, G, and C,
# with a minimum length of 30 and a maximum length of 1000.
# A{5,10}: Matches 5 to 10 consecutive adenine (A) nucleotides,
# which may represent a polyadenylation signal.
# $: the end of the string.

#QUESTION 8

dna = "ATCGCGYAATTCAC"
b=['A','T','G','C']
amb = False
for i,j in enumerate(dna):
    if j not in b :
        amb = True
        print("ambiguous")
if not amb:
    print("not ambiguous")

#QUESTION 9

scientific_name =  "Drosophila melanogaster"
#?P<name> is used to create a named capturing group.
# This allows you to assign a name to a specific group within your regular expression.
#?P: This is the syntax to start a named capturing group.
#<name>: This is the name you give to the capturing group. It can be any valid Python identifier.

pattern = re.compile('(?P<genus>[A-Z][a-z]+)\s+(?P<species>[a-z]+)')
a = pattern.match(scientific_name)
if a:
    genus = a.group('genus')
    species = a.group('species')
    print(f"Genus: {genus}")
    print(f"Species: {species}")
else:
    print("Invalid scientific name format.")


#QUESTION 10
dna = "CGATNCGGAACGATC"
nucleotides = ['A', 'T', 'G', 'C']

for i, base in enumerate(dna):
    if base not in nucleotides:
        print(f"Ambiguous base '{base}' found at position {i}")

#QUESION 11
import re

dna = "ACTGCATTATATCGTACGAAATTATACGCGCG"
pattern = re.compile('TAT+')
a = pattern.findall(dna)
print(a)

#QUESTION 12
dna = "ACTNGCATRGCTACGTYACGATSCGAWTCG"
b=['A','T','G','C']
new_dna=[]
a=0
for i,j in enumerate(dna):
    if j not in b:
        new_dna.append(dna[a:i])
        a=i+1
print(new_dna)

#QUESTION 13
accessions = ['xkn59438', 'yhdck2', 'eihd39d9', 'chdsye847', 'hedle3455', 'xjhd53e', '45da', 'de37dp']

# contain the number 5
# contain the letter d or e
# contain the letters d and e in that order
# contain the letters d and e in that order with a single letter between them
# contain both the letters d and e in any order
# start with x or y
# start with x or y and end with e
# contain three or more digits in a row
# end with d followed by either a, r or p

rule1 = [i for i in accessions if re.search('5', i)]
rule2 = [i for i in accessions if re.search('[de]', i)]
rule3 = [i for i in accessions if re.search('de', i)]
rule4 = [i for i in accessions if re.search('d.e', i)]
rule5 = [i for i in accessions if re.search('d.*e|e.*d', i)]
rule6 = [i for i in accessions if re.search('^[xy]', i)]
rule7 = [i for i in accessions if re.search('^[xy].*e$', i)]
rule8 = [i for i in accessions if re.search('\d{3,}', i)]
rule9 = [i for i in accessions if re.search(r'd[arp]$', i)]

# Print results
print("Rule 1: ",rule1)
print("Rule 2: ",rule2)
print("Rule 3: ",rule3)
print("Rule 4: ",rule4)
print("Rule 5: ",rule5)
print("Rule 6: ",rule6)
print("Rule 7: ",rule7)
print("Rule 8: ",rule8)
print("Rule 9: ",rule9)
