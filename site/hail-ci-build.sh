#!/bin/bash

set -ex

nginx -t -c hail.nginx.conf
