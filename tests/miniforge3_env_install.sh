#!/bin/bash

module purge

module load miniforge3

mamba create -n env_for_test numpy pandas

mamba activate env_for_test

mamba install jax

mamba list -n env_for_test

mamba deactivate 

mamba remove -n env_for_test --all

module unload miniforge3
