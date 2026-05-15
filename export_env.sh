#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/lecarb
export $(grep -v '^#' .env | xargs)