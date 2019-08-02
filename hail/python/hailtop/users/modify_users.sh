#!/bin/sh

op=$1

while read user; do
    stringarray=($user)
    email=${stringarray[0]}
    auth0_id=${stringarray[1]}
    is_developer=${stringarray[2]}
    is_sa=${stringarray[3]}
    namespace=${stringarray[4]}
    echo "email $email"
    echo "auth0_id $auth0_id"
    echo "is_developer $is_developer"
    echo "is_sa $is_sa"
    echo "namespace $namespace"
    make user email="$email" user_id="$auth0_id" is_dev=$is_developer is_sa=$is_sa namespace="$namespace" op=$op
done <users.txt