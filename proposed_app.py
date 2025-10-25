# -*- coding: utf-8 -*-
"""
Main entry point for SnuggleBots - PLP-RAG System
This file serves as the main module for Streamlit deployment
"""

import streamlit as st
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the main unified app
from apps.unified_petbot_app import main

# Run the application
if __name__ == "__main__":
    main()
