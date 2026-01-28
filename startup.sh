#!/bin/bash
cd /testbed/ml-explainability-pack
uvicorn app.main:app --host 0.0.0.0 --port 8000