#!/usr/bin/env python3
"""
AI-Guided Instruction Set Extension for RISC-V
Streamlit Dashboard Application
"""

import os
import sys
from pathlib import Path

# Add project directories to path
sys.path.append(str(Path(__file__).parent))

# Import dashboard application
from dashboard.app import main

# Run the Streamlit dashboard
if __name__ == "__main__":
    main()
