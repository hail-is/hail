#!/bin/bash
set -ex

MASTER=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/master")
TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/token")

curl -X POST -d "{\"token\":\"$TOKEN\"}" http://$MASTER:5000/shutdown
