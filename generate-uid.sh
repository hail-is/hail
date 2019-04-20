#!/bin/sh
LC_CTYPE=C LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c ${1:-8}
