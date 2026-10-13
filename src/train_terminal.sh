#!/usr/bin/env bash

set -euo pipefail

echo "[ ] Training terminal cost..."

python3 scripts/train_safety_value.py --data-dir logs/safety_value_rollouts/ --mode direct --output  safety_filter/quad/safety_value_checkpoint/direct.msgpack

python3 scripts/train_safety_value.py --data-dir logs/safety_value_rollouts/ --mode residual --output  safety_filter/quad/safety_value_checkpoint/residual.msgpack

echo "[ ] All done."