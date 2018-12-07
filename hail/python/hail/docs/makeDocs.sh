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

echo $(find www -name \*.md)
for f in $(find www -name \*.md)
do
    base=$(basename $f | sed 's/\.md//')
    pandoc -s $f \
           -f markdown \
           -t html \
           --mathjax \
           --highlight-style=pygments \
           --columns 10000 \
        | xsltproc -o build/www/$base.html --html www/$base.xslt -
done

# sed for creating GitHub links
pandoc <(cat python/hail/docs/change_log.md | sed -E "s/\(hail\#([0-9]+)\)/(\[#\1](https:\/\/github.com\/hail-is\/hail\/pull\/\1))/g") -o python/hail/docs/change_log.rst

cp -R python build/tmp

(cd build/tmp/python/hail/docs && make BUILDDIR=_build clean html)

DEST="build/www/docs/${HAIL_SHORT_VERSION}/"
rm -rf ${DEST}
mv build/tmp/python/hail/docs/_build/html $DEST
