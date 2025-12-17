#!/bin/bash

# Stock Prediction Web App Startup Script
# ========================================
# This script starts both the backend and frontend servers

set -e

echo "=========================================="
echo " Stock Prediction Web App"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo -e "${BLUE}Checking requirements...${NC}"

if ! command_exists python3; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

if ! command_exists node; then
    echo "Error: Node.js is required but not installed."
    exit 1
fi

if ! command_exists npm; then
    echo "Error: npm is required but not installed."
    exit 1
fi

echo -e "${GREEN}✓ All requirements met${NC}"
echo ""

# Install backend dependencies if needed
echo -e "${BLUE}Setting up backend...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing backend dependencies..."
pip install -q -r backend/requirements.txt

# Download NLTK data if needed
python3 -c "import nltk; nltk.download('vader_lexicon', quiet=True)" 2>/dev/null || true

echo -e "${GREEN}✓ Backend ready${NC}"
echo ""

# Install frontend dependencies if needed
echo -e "${BLUE}Setting up frontend...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo -e "${GREEN}✓ Frontend ready${NC}"
echo ""

cd ..

# Start servers
echo "=========================================="
echo -e "${GREEN}Starting servers...${NC}"
echo "=========================================="
echo ""

# Start backend in background
echo -e "${BLUE}Starting FastAPI backend on http://localhost:8000${NC}"
cd backend
source ../venv/bin/activate
PYTHONPATH=../src uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo -e "${BLUE}Starting Next.js frontend on http://localhost:3000${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "=========================================="
echo -e "${GREEN}Web app is running!${NC}"
echo "=========================================="
echo ""
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Handle shutdown
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait

