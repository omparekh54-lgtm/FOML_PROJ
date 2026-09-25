#!/usr/bin/env bash
set -o errexit
set -o pipefail

pip install --upgrade pip
pip install -r requirements.txt

python -m ml.training.build_dataset --source .
python -m ml.training.train_body_part_router
python -m ml.training.train_abnormality_classifier --body-part chest
python -m ml.training.train_abnormality_classifier --body-part bone

rm -rf datasets
python manage.py collectstatic --no-input
python manage.py migrate --no-input
