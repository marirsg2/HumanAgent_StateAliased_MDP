import os 


directory = "Results"

all_files = os.listdir("./"+directory+"/Raw_output")
FILENAMES =[]
for f in all_files :
    if("policy" in f) : 
        FILENAMES.append(f)


print (FILENAMES)

results = []


for file_name in FILENAMES :

    # file_name = "policy_simplif_results_gridworld20210530-105027_NUM_STATES_100.txt"#great results set
    source_file = "./"+directory+ "/"+ file_name

    with open(source_file,"r") as src:
        all_lines = src.read()

    Bnb_file = "./"+directory+"/Raw_output/" + file_name

    with open(Bnb_file, "r") as f : 
        bnb_file = f.read()

    bnb_file = bnb_file.split("BRANCH AND BOUND")[1]


    HC = all_lines.split("trial_idx")
    params = HC[0]

    GAMMA = float(params.split("DISCOUNT_FACTOR")[1].split("\n")[0].split("=")[1].strip())
    RR = float(params.split("REWARD_NOISE_RANGE")[1].split("\n")[0].split("=")[1].strip())
    RHO = float(params.split("PROB_OF_RANDOM_ACTION")[1].split("\n")[0].split("=")[1].strip())

    HC = HC[1:]

    HC_VALUES = []
    for hc in HC :
        # trial_idx = int(hc.split("\n")[0].strip().split("=")[1])
        expected_value = float(hc.split("expected_value")[1].split("\n")[0].split("=")[1].strip())
        HC_VALUES.append(expected_value)



    # BNB STUFF ------
    bnb_file = bnb_file.split("solver results")[1]

    NODES = int(bnb_file.split("- nodes:")[1].split("\n")[0].strip())
    BNB_VALUE = float(bnb_file.split("Expected Value with HASA")[1].split("\n")[0].split("=")[1].split(",")[0].replace("tensor(","").strip())


    # print (RHO, GAMMA, RR)
    # print (HC_VALUES)
    # print (NODES)
    # print (BNB_VALUE)

    tmp = {}
    tmp['RR'] = RR
    tmp['RHO'] = RHO
    tmp['GAMMA'] = GAMMA
    tmp['HC_VALUES'] = HC_VALUES
    tmp['NODES'] = NODES
    tmp['BNB_VALUE'] = BNB_VALUE

    results.append(tmp)





from pprint import pprint
pprint (results)


from matplotlib import pyplot as plt
import numpy as np

# PLOTS -----------
domain_suffix = "gridworld"
dest_folder = "./Results/Plots"

# GAMMA
# RR = 0 
# RHO = 0.5

#BOXPLOT GAMMA v Value
fig1 = plt.figure()
ax= fig1.add_subplot(1,1,1)

x_axis = [0.3, 0.5, 0.7, 0.9]

HC_Value_data = []
bnb_value_y_axis = []

for gamma in x_axis :
    for r in results : 
        if gamma == r['GAMMA'] and r['RHO'] == 0.05 and r['RR'] == 0 : 
            hcval = np.array(r['HC_VALUES'])
            bnboptima = r['BNB_VALUE']

            hcval /= bnboptima

            HC_Value_data.append(hcval)
            bnb_value_y_axis.append(bnboptima)



print (len(HC_Value_data))


ax.boxplot(HC_Value_data,showfliers =True)
# ax.scatter(range(1,len(x_axis)+1),bnb_value_y_axis)

plt.xticks(range(1,len(x_axis)+1), x_axis)
ax.set_xlabel(r'Discount Factor')
ax.set_ylabel(r'Value')
ax.set_title(r'Discount Factor vs Value')
fig1.savefig(dest_folder+"/boxplot_gamma_v"+domain_suffix+".png")








# RR v VALUE
# GAMMA = 0.7
# RHO = 0.5

#BOXPLOT GAMMA v Value
fig1 = plt.figure()
ax= fig1.add_subplot(1,1,1)

x_axis = [0, 1, 2, 4]

HC_Value_data = []
bnb_value_y_axis = []

for val in x_axis :
    for r in results : 
        if val == r['RR'] and r['RHO'] == 0.05 and r['GAMMA'] == 0.7 : 
            hcval = np.array(r['HC_VALUES'])
            bnboptima = r['BNB_VALUE']

            hcval /= bnboptima

            HC_Value_data.append(hcval)
            bnb_value_y_axis.append(bnboptima)



print (len(HC_Value_data))


ax.boxplot(HC_Value_data,showfliers =True)
# ax.scatter(range(1, len(x_axis)+1), [1,1,1,1])
# ax.scatter(range(1,len(x_axis)+1),bnb_value_y_axis)

plt.xticks(range(1,len(x_axis)+1), x_axis)
ax.set_xlabel(r'Reward Noise Range')
ax.set_ylabel(r'Value')
ax.set_title(r'Reward Noise Range vs Value')
fig1.savefig(dest_folder+"/boxplot_rr_v"+domain_suffix+".png")

















