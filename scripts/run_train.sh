#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 3 ./tools/train.py
