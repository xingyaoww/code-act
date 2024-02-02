#!/bin/bash
DB_DIR=`pwd`/data/mongodb
# Make sure the directory exists
if [ ! -d "$DB_DIR" ]; then
    mkdir -p $DB_DIR
fi
echo "Starting mongodb in $DB_DIR"

USERNAME=codeactagent
PASSWORD=$1
echo "USERNAME=$USERNAME"
echo "PASSWORD=$PASSWORD"
docker run \
    --rm \
    -p 27017:27017 \
    -v $DB_DIR:/data/db \
    --user $(id -u):$(id -g) \
    -e MONGODB_INITDB_ROOT_USERNAME=$USERNAME \
    -e MONGODB_INITDB_ROOT_PASSWORD=$PASSWORD \
    --name mongo-chat-ui \
    -d mongodb/mongodb-community-server:latest
