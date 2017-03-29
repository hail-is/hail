#!/bin/sh

echo_build_properties() {
  echo user=$USER
  echo revision=$(git rev-parse HEAD)
  echo branch=$(git rev-parse --abbrev-ref HEAD)
  echo date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo url=$(git config --get remote.origin.url)
}

mkdir -p build/extra-resources

echo_build_properties > "src/main/resources/build-info.properties"
