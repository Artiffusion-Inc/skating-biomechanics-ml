#!/bin/bash
cd /root/skating-biomechanics-ml/experiments/rtmpose-simcc-kd
source .venv/bin/activate
python -m tools.train configs/rtmpose_s_coco17_skating.py --launcher none --work-dir work_dirs/rtmpose_s_baseline
