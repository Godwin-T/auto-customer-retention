#!/bin/bash

set -e  # Exit on any error

echo "📁 Ensuring ./databases directory exists..."
mkdir -p ./databases

echo "🧠 Running Alembic migrations..."
alembic upgrade head

echo "✅ Database is ready!"
