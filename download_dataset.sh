#!/bin/bash
# Download the dataset
# to verify the signature, run
# Returns 0 if match, 1 if different
# shasum leaves.zip | cut -d' ' -f1 | grep -q "27fe761aa373e687ac3259a67b9b144e63c70cd5" && echo "MATCH" || echo "NO MATCH"

curl -o leaves.zip https://cdn.intra.42.fr/document/document/17547/leaves.zip
mkdir -p images
unzip leaves.zip -d temp_unzip
mv temp_unzip/images/* images/
rm -rf temp_unzip
#rm leaves.zip
