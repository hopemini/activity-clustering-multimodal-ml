#!/bin/bash

export PYTHONPATH=.
python evaluation.py -e purity
python evaluation.py -e nmi
python evaluation.py -e ari
echo 'Done...'
