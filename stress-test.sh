#!/bin/bash

wrk \
  -t 2 \
  -c 2 \
  -d 30s \
  --latency \
  -H 'accept: application/json' \
  'http://localhost:8000/batch-predict?k=5'
