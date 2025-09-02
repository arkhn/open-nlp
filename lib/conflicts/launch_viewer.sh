#!/bin/bash

# Healthcare Data Viewer Launcher
# This script launches the Streamlit web interface for healthcare domain experts

echo "🏥 Starting Clinical Conflict Data Viewer..."
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "clinical_data_viewer.py" ]; then
    echo "❌ Error: clinical_data_viewer.py not found in current directory"
    echo "Please run this script from the lib/conflicts directory"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python -c "import streamlit, pandas, plotly, numpy, pyarrow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip install -r requirements.txt
fi

# Launch the application
echo "🚀 Launching web interface..."
echo "The application will open in your default web browser"
echo "If it doesn't open automatically, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run clinical_data_viewer.py --server.port 8501 --server.address localhost
