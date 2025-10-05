#!/usr/bin/env python3
"""
Initialize the database with corpus data
"""

import sys
import os
from pathlib import Path

# Add the service directory to the Python path
service_dir = Path(__file__).parent
sys.path.insert(0, str(service_dir))

from database import create_tables, SessionLocal
from data_loader import DataLoader

def main():
    print("Initializing CNE Word Database...")
    
    # Create tables
    print("Creating database tables...")
    create_tables()
    print("✓ Database tables created")
    
    # Load corpus data
    print("Loading corpus data...")
    corpus_path = "/Volumes/Storage/CNE/data/cne_corpus"
    
    if not os.path.exists(corpus_path):
        print(f"❌ Error: Corpus path not found: {corpus_path}")
        return
    
    db = SessionLocal()
    try:
        loader = DataLoader(db)
        loader.load_corpus(corpus_path)
        print("✓ Corpus data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading corpus data: {e}")
    finally:
        db.close()
    
    print("Database initialization complete!")

if __name__ == "__main__":
    main()

