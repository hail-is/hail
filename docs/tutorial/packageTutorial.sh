#!/bin/bash

###################################################################
# Converts markdown version of tutorial to iPython notebook
# Copies images for tutorial to directory
# Package into tarball
###################################################################

set -e

markdownFile=$1
imageDirectory=$2
tmpDirectory=$3
outputDirectory=$4

notedown $markdownFile > ${tmpDirectory}Hail_Tutorial-v1.ipynb #${tmpDirectory}Hail_Tutorial-v1.ipynb.tmp

#sed 's/"metadata": {},/"metadata": {"anaconda-cloud": {}, "kernelspec": {"display_name": "Bash", "language": "bash", "name": "bash" }},/g' ${tmpDirectory}Hail_Tutorial-v1.ipynb.tmp > build/tmp/tutorial/Hail_Tutorial-v1.ipynb

#rm ${tmpDirectory}Hail_Tutorial-v1.ipynb.tmp
cp ${imageDirectory}*.png ${tmpDirectory}
tar -cvf ${outputDirectory}Hail_Tutorial-v1.tgz -C ${tmpDirectory} .