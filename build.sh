#!/usr/bin/env bash
# Render build step: install, assemble the dataset from the raw Kaggle folders
# in this repo, train every model, then prepare static files and the database.
set -o errexit
set -o pipefail

pip install --upgrade pip
pip install -r requirements.txt

python -m ml.training.build_dataset --source .
python -m ml.training.train_body_part_router
python -m ml.training.train_abnormality_classifier --body-part chest
python -m ml.training.train_abnormality_classifier --body-part bone

# The assembled training copies aren't needed at runtime.
rm -rf datasets

python manage.py collectstatic --no-input
python manage.py migrate --no-input
