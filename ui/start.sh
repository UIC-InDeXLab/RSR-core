#!/bin/bash
# Start both the FastAPI backend and Vite frontend dev servers.
# Usage: ./ui/start.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting RSR-core UI..."
echo "  Backend:  http://localhost:8042"
echo "  Frontend: http://localhost:5173"
echo ""

# Start backend
cd "$PROJECT_ROOT"
python -m uvicorn ui.backend.main:app --host 0.0.0.0 --port 8042 --reload &
BACKEND_PID=$!

# Start frontend
cd "$SCRIPT_DIR/frontend"
npx vite --host &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

echo ""
echo "Press Ctrl+C to stop both servers."
wait
