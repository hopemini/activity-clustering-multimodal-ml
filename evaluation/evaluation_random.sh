#!/bin/bash

export PYTHONPATH=.
python evaluation_random.py -e purity
python evaluation_random.py -e nmi
python evaluation_random.py -e ari
echo 'Done...'
