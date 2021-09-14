echo "Running Experiments..."

PROB_OF_RANDOM_ACTION=$1
DISCOUNT=$2
REWARD_NOISE_RANGE=$3



# Fix the discount & stochasticity here 
OS=`uname`
if [ "$OS" = 'Darwin' ]; then
	echo "Running on MacOS"
	gsed -i -e '60d' 'src/Combined_loss/config.py'
	gsed -i '60i DISCOUNT_FACTOR ='$DISCOUNT 'src/Combined_loss/config.py'
	gsed -i -e '71d' 'src/Combined_loss/config.py'
	gsed -i '71i PROB_OF_RANDOM_ACTION = '$PROB_OF_RANDOM_ACTION 'src/Combined_loss/config.py'
else
    # for Linux and Windows
    echo "Running on Linux/Windows, check if config.py is changing correctly."
	sed -i -e '60d' 'src/Combined_loss/config.py'
	sed -i '60i DISCOUNT_FACTOR ='$DISCOUNT 'src/Combined_loss/config.py'
	sed -i -e '71d' 'src/Combined_loss/config.py'
	sed -i '71i PROB_OF_RANDOM_ACTION = '$PROB_OF_RANDOM_ACTION 'src/Combined_loss/config.py'
fi


python -m src.SAMDP_HillClimbing_policy_iteration_runner --rr $REWARD_NOISE_RANGE