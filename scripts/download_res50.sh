#!/usr/bin/env bash

mkdir model
pushd model
wget http://data.dmlc.ml/mxnet/models/imagenet/resnet/50-layers/resnet-50-0000.params
popd