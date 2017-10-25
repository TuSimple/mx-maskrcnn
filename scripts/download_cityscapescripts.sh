#!/usr/bin/env bash

pushd data/cityscape/
git clone https://github.com/mcordts/cityscapesScripts

mkdir -p results/pred/

cd cityscapesScripts

python setup.py build_ext --inplace

mv cityscapesscripts ../

popd