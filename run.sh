DBNAME=$DBNAME
MYSQL_USERNAME=$MYSQL_USERNAME
MYSQL_PASSWORD=$MYSQL_PASSWORD
THOSTNAME=$THOSTNAME

# cat <<EOT > .env
# DBNAME=$DBNAME
# MYSQL_USERNAME=$MYSQL_USERNAME
# MYSQL_PASSWORD=$MYSQL_PASSWORD
# HOSTNAME=$THOSTNAME
# EOT

# echo "MySQL Username: $MYSQL_USERNAME"
# echo "MySQL Password: $MYSQL_PASSWORD"
# echo "Hostname: $THOSTNAME"
# echo "DB Name: $DBNAME"

# Example usage
mysql -h $THOSTNAME -u $MYSQL_USERNAME -p$MYSQL_PASSWORD -e "SHOW DATABASES;"


python_output=$(python k.py)

echo $THOSTNAME
echo $MYSQL_PASSWORD
echo $MYSQL_USERNAME
echo $DBNAME

# Set the output of the Python script as an environment variable
DEPLOY=$python_output
echo $DEPLOY
