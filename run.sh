echo '=================================================================================='
python_output=$(python run.py)
echo '=================================================================================='
# Set the output of the Python script as an environment variable
DEPLOY=$python_output
echo $DEPLOY
