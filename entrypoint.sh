#!/bin/bash

# Start FastAPI
uvicorn server.main:app --host 0.0.0.0 --port 8000

# Exit with status of process that exited
exit $?
