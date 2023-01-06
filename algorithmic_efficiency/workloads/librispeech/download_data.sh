#! /bin/bash

set -x

for d in dev test; do
    for s in clean other; do
	echo $d, $s
	wget --quiet --progress=bar:force:noscroll http://www.openslr.org/resources/12/$d-$s.tar.gz
	tar xzf $d-$s.tar.gz
    done
done

wget --quiet --progress=bar:force:noscroll http://www.openslr.org/resources/12/raw-metadata.tar.gz
wget --quiet --progress=bar:force:noscroll http://www.openslr.org/resources/12/train-clean-100.tar.gz

# Untar files
tar xzf raw-metadata.tar.gz
tar xzf train-clean-100.tar.gz

