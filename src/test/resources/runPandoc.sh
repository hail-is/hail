#!/bin/bash

###################################################################
# Takes and input directory and converts all markdown files to HTML 
# with the same directory structure in the parameter given by output 
# directory
###################################################################

set -ex

inDir=$1 # input directory with markdown files
outDir=$2 # output directory to put html files in

mkdir -p $outDir

for mdFile in $(find $inDir -name '*.md'); do
    mkdir -p $outDir$(dirname $mdFile)
    pandoc $mdFile -f markdown -t html --mathjax --highlight-style=pygments --columns 10000 -o $outDir${mdFile/%md/html}
done
