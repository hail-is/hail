#!/bin/bash

set -ex

cleanup() {
    trap "" INT TERM
    set +e
    rm -f python/hail/docs/change_log.rst
}
trap cleanup EXIT
trap 'exit 1' INT TERM

# make directories
mkdir -p build/www/ build/tmp/python/ build/tmp/docs build/www/docs

# copy website content
cp www/*.{js,css,css.map,html,png} build/www #  www/annotationdb/* does not exist

pandoc -s www/landing.md -f markdown -t html --mathjax --highlight-style=pygments --columns 10000 -o build/tmp/landing.html
xsltproc --html -o build/www/index.html www/readme-to-index.xslt build/tmp/landing.html

pandoc -s www/jobs.md -f markdown -t html --mathjax --highlight-style=pygments --columns 10000 -o build/tmp/jobs.html
xsltproc --html -o build/www/jobs.html www/jobs.xslt build/tmp/jobs.html

pandoc -s www/about.md -f markdown -t html --mathjax --highlight-style=pygments --columns 10000 -o build/tmp/about.html
xsltproc --html -o build/www/about.html www/about.xslt build/tmp/about.html

# sed for creating GitHub links
pandoc <(cat python/hail/docs/change_log.md | sed -E "s/\(hail\#([0-9]+)\)/(\[#\1](https:\/\/github.com\/hail-is\/hail\/pull\/\1))/g") -o python/hail/docs/change_log.rst

cp -R python build/tmp

(cd build/tmp/python/hail/docs && make BUILDDIR=_build clean html)

DEST="build/www/docs/${HAIL_SHORT_VERSION}/"
rm -rf ${DEST}
mv build/tmp/python/hail/docs/_build/html $DEST
