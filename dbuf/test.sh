set -ex

id=$(curl -sSL -XPOST localhost:5000/)
key1=$(curl -sSL -XPOST localhost:5000/$id -d 'fuckfuckfuck0')
key2=$(curl -sSL -XPOST localhost:5000/$id -d 'fuckfuckfuck1')
key3=$(curl -sSL -XPOST localhost:5000/$id -d 'fuckfuckfuck2')
key4=$(curl -sSL -XPOST localhost:5000/$id -d 'fuckfuckfuck3')
key5=$(curl -sSL -XPOST localhost:5000/$id -d 'fuckfuckfuck4')
curl -sSL -XGET localhost:5000/$id -d "$key1"
curl -sSL -XGET localhost:5000/$id -d "$key1"
curl -sSL -XGET localhost:5000/$id -d "$key3"
curl -sSL -XGET localhost:5000/$id -d "$key5"
curl -sSL -XGET localhost:5000/$id -d "$key5"
curl -sSL -XDELETE localhost:5000/$id
