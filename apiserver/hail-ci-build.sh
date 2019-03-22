#!/bin/bash
set -ex

time source activate hail

time make test
