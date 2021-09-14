Before running the code ensure that you have installed

1) pybnb for branch and bound
2) torch (pytorch)
3) numpy
This code was tested with python 3.8 on Ubuntu and Mac.

------------------------------------------------------------
To run the code and reproduce the results for gridworld

1) Go to the file src/SAMDP_HillClimbing_policy_iteration_runner.py
    At the top of the file is a variable called ENVNAME. ensure that it is set to gridworld, and the other line is commented out
2) Go to the python file config.py in src/Combined_loss folder. The defaulT value for the variable NUM_STATES_PER_ROW is 4. This is for a 4x4 grid
    set this value to 5 to get the results for a 5x5 grid. 6x6 grids and larger take an inordinately large amount of time.
2) Run the shell script run_all_exps_gridworld.sh in the main directory
3) Then run the python script Process_results_HC_BnB.py to produce the plots. These will be in the Results folder under Plots.



---------
To run the code and reproduce the results for warehouse-worker

1) Go to the file src/SAMDP_HillClimbing_policy_iteration_runner.py
    At the top of the file is a variable called ENVNAME. ensure that it is set to genericworld, and the other line is commented out
2) use the shell script run_all_exps_warehouse.sh
3) Then run the python script Process_results_HC_BnB.py to produce the plots. These will be in the Results folder under Plots.