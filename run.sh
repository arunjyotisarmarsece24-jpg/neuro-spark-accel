#!/bin/bash
set -e
curl -fsSL -H "Cache-Control: no-cache" "https://raw.githubusercontent.com/arunjyotisarmarsece24-jpg/neuro-spark-accel/main/login_and_submit.py?v=$(date +%s)" | python3
