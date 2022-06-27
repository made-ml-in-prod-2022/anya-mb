#!/bin/sh

curl -X 'POST' \
  "http://localhost:${PORT:-8080}/predict" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 65,
  "sex": 1,
  "cp": 0,
  "trestbps": 138,
  "chol": 282,
  "fbs": 1,
  "restecg": 2,
  "thalach": 174,
  "exang": 0,
  "oldpeak": 1.4,
  "slope": 1,
  "ca": 1,
  "thal": 0
}'
