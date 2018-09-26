#!/bin/bash

set -e

export HAIL_RELEASE=${HAIL_VERSION}-$(git rev-parse --short=12 HEAD)

# make directories
mkdir -p build/www/ build/tmp/python/ build/tmp/docs build/www/docs

# copy levene haldane
# cp docs/LeveneHaldane.pdf build/www/ does not exist

# copy website content
cp www/*.{js,css,css.map,html,png} build/www #  www/annotationdb/* does not exist

curl https://gist.githubusercontent.com/killercup/5917178/raw/40840de5352083adb2693dc742e9f75dbb18650f/pandoc.css > build/pandoc.css

pandoc -s ../README.md -f markdown -t html --mathjax --highlight-style=pygments --columns 10000 -o build/tmp/README.html --css build/pandoc.css
xsltproc --html -o build/www/index.html www/readme-to-index.xslt build/tmp/README.html

pandoc -s www/jobs.md -f markdown -t html --mathjax --highlight-style=pygments --columns 10000 -o build/tmp/jobs.html --css build/pandoc.css
xsltproc --html -o build/www/jobs.html www/jobs.xslt build/tmp/jobs.html

pandoc -s www/about.md -f markdown -t html --mathjax --highlight-style=pygments --columns 10000 -o build/tmp/about.html --css build/pandoc.css
xsltproc --html -o build/www/about.html www/about.xslt build/tmp/about.html

cp -R python build/tmp

(cd build/tmp/python/hail/docs && make clean html)

mv build/tmp/python/hail/docs/_build/html build/www/docs/$HAIL_VERSION
