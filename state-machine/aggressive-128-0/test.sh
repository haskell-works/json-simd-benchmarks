#!/usr/bin/env bash

echo "====="
echo "Running state-machine/aggressive-128-0"
time cat ../../data/250m.json   | ./a.out /dev/stdin 250m.json.ib.idx 250m.json.bp.idx 250m.json.op.idx 250m.json.cl.idx
time cat ../../data/simple.json | ./a.out /dev/stdin simple.json.ib.idx simple.json.bp.idx simple.json.op.idx simple.json.cl.idx
time cat ../../data/1024.json   | ./a.out /dev/stdin 1024.json.ib.idx 1024.json.bp.idx 1024.json.op.idx 1024.json.cl.idx
time cat ../../data/64k.json   | ./a.out /dev/stdin 64k.json.ib.idx 64k.json.bp.idx 64k.json.op.idx 64k.json.cl.idx
