#!/bin/bash
JUPYTER_API_PORT=$1
echo "JUPYTER_API_PORT=$JUPYTER_API_PORT"

pushd scripts/chat/code_execution
export PYTHONPATH=`pwd`:$PYTHONPATH
# gunicorn -w 1 api:app --bind localhost:$JUPYTER_API_PORT
# TODO: fix the issue of sharing data across worker before enabling multiple workers
python3 api.py --port $JUPYTER_API_PORT

popd
