#!/bin/bash -e

cd $(dirname $0)

git submodule sync
git submodule update --init gdrcopy
git submodule update --init gtest
git submodule update --init nccl
git submodule update --init ucx
