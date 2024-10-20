DBNAME=$DBNAME
MYSQL_USERNAME=$MYSQL_USERNAME
MYSQL_PASSWORD=$MYSQL_PASSWORD
HOSTNAME=$HOSTNAME

# cat <<EOT > .env
# DBNAME=$DBNAME
# MYSQL_USERNAME=$MYSQL_USERNAME
# MYSQL_PASSWORD=$MYSQL_PASSWORD
# HOSTNAME=$HOSTNAME
# EOT

echo "MySQL Username: $MYSQL_USERNAME"
echo "MySQL Password: $MYSQL_PASSWORD"
echo "Hostname: $HOSTNAME"
echo "DB Name: $DBNAME"

ping -c 4 $HOSTNAME

# Example usage
mysql -h $HOSTNAME -u $MYSQL_USERNAME -p$MYSQL_PASSWORD -e "SHOW DATABASES;"


python_output=$(python k.py)

echo $HOSTNAME
echo $MYSQL_PASSWORD
echo $MYSQL_USERNAME
echo $DBNAME

# Set the output of the Python script as an environment variable
DEPLOY=$python_output
echo $DEPLOY
