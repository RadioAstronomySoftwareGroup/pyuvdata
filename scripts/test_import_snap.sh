#!/bin/bash
source activate hera_pspec_testing
python import_snap.py -c ../pyuvdata/data/snap_correlation_1526493089.yaml -z ../pyuvdata/data/snap_correlation_1526493089.npz
