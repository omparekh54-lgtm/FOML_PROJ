#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -r requirements.txt
python ml/training/build_dataset.py
python manage.py migrate --noinput
