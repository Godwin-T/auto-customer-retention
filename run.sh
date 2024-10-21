# Example usage
mysql -h $HOSTNAME -u $MYSQL_USERNAME -p$MYSQL_PASSWORD -e "SHOW DATABASES;"

echo '=================================================================================='


python_output=$(python k.py)


echo '================================================================='
# Set the output of the Python script as an environment variable
DEPLOY=$python_output
echo $DEPLOY
