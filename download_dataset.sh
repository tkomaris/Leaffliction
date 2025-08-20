#!/bin/bash
# Download the dataset

curl -o leaves.zip https://cdn.intra.42.fr/document/document/17547/leaves.zip
mkdir -p images
unzip leaves.zip -d temp_unzip
mv temp_unzip/images/* images/
rm -rf temp_unzip
rm leaves.zip