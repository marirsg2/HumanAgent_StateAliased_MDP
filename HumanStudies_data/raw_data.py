import numpy as np
import os
from scipy import stats

source_data_file = "./HumanStudiesData_Jan18.txt"


with open(source_data_file,"r") as src:
    all_lines = src.readlines()

data_lines = []
for line in all_lines:
    if "ImmutableMultiDict" in line:
        data_lines.append(line)
#--end for loop

data_list = []
for line in data_lines:
    temp_line = line.replace("ImmutableMultiDict(","").replace(")\n","")
    temp_line = "temp_data = " + temp_line
    exec (temp_line)
    temp_dict = {x[0]:x[1] for x in temp_data }
    if temp_dict["tid"].startswith("A"): #valid turker IDs start with "A"
        data_list.append(temp_dict)
#--

tid = [x["tid"] for x in data_list]


complex_policy_correct = np.array([int(x["game1_correct"]) for x in data_list])
complex_policy_wrong = np.array([int(x["game1_wrong"]) for x in data_list])
complex_policy_attempts = np.array([int(x["game1_attempts"]) for x in data_list])
complex_policy_mean = np.mean(complex_policy_correct/complex_policy_attempts)
complex_policy_std = np.std(complex_policy_correct/complex_policy_attempts)

simple_policy_correct = np.array([int(x["game2_correct"]) for x in data_list])
simple_policy_wrong = np.array([int(x["game2_wrong"]) for x in data_list])
simple_policy_attempts = np.array([int(x["game2_attempts"]) for x in data_list])
simple_policy_mean = np.mean(complex_policy_correct/complex_policy_attempts)
simple_policy_std = np.std(simple_policy_correct/simple_policy_attempts)

def myprint(l) : 
    for ix in l : 
        print (ix)

# myprint (complex_policy_correct)
# myprint (complex_policy_wrong)
# myprint (complex_policy_attempts)



print ("*"*10)

# myprint (simple_policy_correct)
# myprint (simple_policy_wrong)
# myprint (simple_policy_attempts)


# Calculator used : 
# https://www.socscistatistics.com/tests/studentttest/default2.aspx

# Setting : 
# Two tailed test.

# Complex v Simple Correct 
# The t-value is -7.83825. The p-value is < .00001. The result is significant at p < .05.

# Complex v Simple wrong : 
# The t-value is -0.12418. The p-value is .901486. The result is not significant at p < .05.

# Comlex v Simple Total Attempts
# The t-value is -7.38548. The p-value is < .00001. The result is significant at p < .05.

