#!/bin/bash
cd "$(dirname "$0")"
git log |head -3 > version.txt
source venv/bin/activate
python main.py
# exit 0

