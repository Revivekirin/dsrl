#!/bin/bash

conda init
conda activate dsrl

set -e

echo "===== Start Pipeline ====="

python train_dsrl_fql.py --config-path=cfg/robomimic/fql --config-name=dsrl_can.yaml

python train_dsrl_fql.py --config-path=cfg/robomimic/fql --config-name=dsrl_can_offline.yaml

python train_dsrl_fql.py --config-path=cfg/robomimic/fql --config-name=dsrl_can_online.yaml

echo "===== All Done! ====="

