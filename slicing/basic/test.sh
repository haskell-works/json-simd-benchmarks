#!/usr/bin/env bash

echo "====="
echo "Running slicing/basic"
time cat ../../data/250m.json | ./a.out /dev/stdin 250m.json.ib.idx 250m.json.bp.idx
