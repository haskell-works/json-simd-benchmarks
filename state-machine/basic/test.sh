#!/usr/bin/env bash

echo "====="
echo "Running state-machine/basic"
time cat ../../data/250m.json | ./a.out /dev/stdin 250m.json.ib.idx 250m.json.bp.idx 250m.json.op.idx 250m.json.cl.idx
