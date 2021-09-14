import numpy as np
import os
from scipy import stats

source_data_file = "./HumanStudies_data/HumanStudiesData_Jan18.txt"


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
    temp_dict = {x[0]:x[1] for x in temp_data } #temp data comes from executing the temp_line
    if temp_dict["tid"].startswith("A"): #valid turker IDs start with "A"
        data_list.append(temp_dict)
#--

tid = [x["tid"] for x in data_list]
print("tid ", tid)

complex_policy_attempts = np.array([int(x["game1_attempts"]) for x in data_list])
complex_policy_correct = np.array([int(x["game1_correct"]) for x in data_list])
complex_policy_wrong = np.array([int(x["game1_wrong"]) for x in data_list])
complex_policy_correct_rate = np.array([int(x["game1_correct"]) for x in data_list])/complex_policy_attempts
complex_policy_wrong_rate = np.array([int(x["game1_wrong"]) for x in data_list])/complex_policy_attempts
complex_policy_mean = np.mean(complex_policy_correct)
complex_policy_std = np.std(complex_policy_correct )
complex_policy_attempts_mean = np.mean(complex_policy_attempts)
complex_policy_attempts_std = np.std(complex_policy_attempts )
print("complex_policy_attempts_mean,complex_policy_attempts_std",complex_policy_attempts_mean,complex_policy_attempts_std)

simple_policy_attempts = np.array([int(x["game2_attempts"]) for x in data_list])
simple_policy_correct = np.array([int(x["game2_correct"]) for x in data_list])
simple_policy_wrong = np.array([int(x["game2_wrong"]) for x in data_list])
simple_policy_correct_rate = np.array([int(x["game2_correct"]) for x in data_list]) / simple_policy_attempts
simple_policy_wrong_rate = np.array([int(x["game2_wrong"]) for x in data_list]) / simple_policy_attempts
simple_policy_mean = np.mean(simple_policy_correct)
simple_policy_std = np.std(simple_policy_correct)
simple_policy_attempts_mean = np.mean(simple_policy_attempts)
simple_policy_attempts_std = np.std(simple_policy_attempts )
print("simple_policy_attempts_mean,simple_policy_attempts_std",simple_policy_attempts_mean,simple_policy_attempts_std)

print("num data points = ", complex_policy_correct_rate.shape[0])
print("-----------------------------")
print("complex_policy_correct = ", np.mean(complex_policy_correct_rate), np.std(complex_policy_correct_rate))
print("complex_policy_wrong = ", np.mean(complex_policy_wrong_rate), np.std(complex_policy_wrong_rate))
print("complex_policy_attempts = ", np.mean(complex_policy_attempts), np.std(complex_policy_attempts))
print("complex_policy_accuracy = ", np.mean(complex_policy_correct_rate / complex_policy_attempts), np.std(complex_policy_correct_rate / complex_policy_attempts))
print("------------------------------------")
print("simple_policy_correct = ", np.mean(simple_policy_correct_rate), np.std(simple_policy_correct_rate))
print("simple_policy_wrong = ", np.mean(simple_policy_wrong_rate), np.std(simple_policy_wrong_rate))
print("simple_policy_attempts = ", np.mean(simple_policy_attempts), np.std(simple_policy_attempts))
print("simple_policy_accuracy = ", np.mean(simple_policy_correct_rate / simple_policy_attempts), np.std(simple_policy_correct_rate / simple_policy_attempts))
print("------------------------------------")

print(stats.ttest_ind(simple_policy_correct, complex_policy_correct, equal_var=False, nan_policy='raise', alternative='greater'))
print(stats.ttest_ind(simple_policy_attempts, complex_policy_attempts, equal_var=False, nan_policy='raise', alternative='greater'))
print(stats.ttest_ind(simple_policy_wrong_rate,complex_policy_wrong_rate, equal_var=False, nan_policy='raise', alternative='less'))
print( stats.ttest_ind(stats.norm.rvs(loc=simple_policy_mean,scale=simple_policy_std,size=500),
    stats.norm.rvs(loc=complex_policy_mean, scale=complex_policy_std, size=500),  equal_var=False, nan_policy='raise'))
