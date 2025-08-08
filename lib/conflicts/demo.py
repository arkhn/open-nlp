#!/usr/bin/env python3
"""
Demo script for the Clinical Document Conflict Pipeline
Shows key features and functionality without requiring API calls
"""

import os
from pathlib import Path

def show_system_overview():
    """Display system overview"""
    print("🏥 Clinical Document Conflict Pipeline Demo")
    print("=" * 60)
    print()
    print("This pipeline implements a three-agent system for clinical document")
    print("conflict generation and validation:")
    print()
    print("1. 🩺 Doctor Agent - Analyzes documents and selects conflict types")
    print("2. ✏️  Editor Agent - Modifies documents to introduce conflicts")  
    print("3. ⚖️  Moderator Agent - Validates and scores the modifications")
    print()

def show_conflict_types():
    """Display available conflict types"""
    from config import CONFLICT_TYPES
    
    print("📋 Available Conflict Types")
    print("-" * 40)
    
    for i, (key, conflict_type) in enumerate(CONFLICT_TYPES.items(), 1):
        print(f"{i}. {conflict_type.name.upper()}")
        print(f"   Description: {conflict_type.description}")
        print(f"   Example: {conflict_type.examples[0]}")
        print()

def show_dataset_info():
    """Show dataset information"""
    try:
        from data_loader import ClinicalDataLoader
        
        print("📊 Dataset Information")
        print("-" * 40)
        
        loader = ClinicalDataLoader()
        stats = loader.get_data_statistics()
        
        print(f"Total documents: {stats['total_documents']:,}")
        print(f"Unique subjects: {stats['unique_subjects']:,}")
        print(f"Document categories: {len(stats['sample_categories'])}")
        
        print("\nTop categories by document count:")
        categories = sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)
        for category, count in categories[:5]:
            print(f"  • {category}: {count:,} documents")
        
        print(f"\nText length statistics:")
        text_stats = stats['text_length_stats']
        print(f"  • Average: {text_stats['mean']:.0f} characters")
        print(f"  • Range: {text_stats['min']:,} - {text_stats['max']:,} characters")
        print()
        
    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        print()

def show_sample_documents():
    """Show sample documents from the dataset"""
    try:
        from data_loader import ClinicalDataLoader
        
        print("📄 Sample Documents")
        print("-" * 40)
        
        loader = ClinicalDataLoader()
        pairs = loader.get_random_document_pairs(count=1)
        
        if pairs:
            pair = pairs[0]
            print(f"Document Pair: {pair.doc1_id} & {pair.doc2_id}")
            print(f"Categories: {pair.category1} & {pair.category2}")
            print()
            
            print("📝 Document 1 (first 300 characters):")
            print(f"   {pair.doc1_text[:300]}...")
            print()
            
            print("📝 Document 2 (first 300 characters):")
            print(f"   {pair.doc2_text[:300]}...")
            print()
            
    except Exception as e:
        print(f"❌ Could not load sample documents: {e}")
        print()

def show_api_requirements():
    """Show API requirements and setup"""
    print("🔑 API Setup Requirements")
    print("-" * 40)
    
    api_key = os.getenv("GROQ_API_KEY")
    
    if api_key:
        print("✅ GROQ_API_KEY is configured")
        print("   The pipeline is ready for full operation!")
        print()
        print("   Try these commands:")
        print("   • python main.py batch --size 2")
        print("   • python main.py test-conflict opposition")
        print("   • python main.py stats")
    else:
        print("❌ GROQ_API_KEY is not set")
        print()
        print("To use the full pipeline, you need a Groq API key:")
        print("1. Sign up at https://console.groq.com/")
        print("2. Get your API key")
        print("3. Set it with: export GROQ_API_KEY='your_key_here'")
        print()
        print("Without API key, you can still:")
        print("• View dataset statistics")
        print("• Browse conflict types")
        print("• Check database status")
    print()

def show_usage_examples():
    """Show usage examples"""
    print("💡 Usage Examples")
    print("-" * 40)
    
    print("Basic batch processing:")
    print("  python main.py batch --size 3")
    print()
    
    print("Process documents from same subject:")
    print("  python main.py batch --size 2 --same-subject")
    print()
    
    print("Filter by document category:")
    print("  python main.py batch --categories 'Discharge summary'")
    print()
    
    print("Test specific conflict types:")
    print("  python main.py test-conflict opposition --count 5")
    print("  python main.py test-conflict anatomical")
    print()
    
    print("View information:")
    print("  python main.py stats")
    print("  python main.py list-conflicts") 
    print("  python main.py db-info")
    print()

def show_database_status():
    """Show database status"""
    try:
        from base import DatabaseManager
        
        print("🗄️  Database Status")
        print("-" * 40)
        
        db_manager = DatabaseManager()
        count = db_manager.get_validated_documents_count()
        db_path = Path(db_manager.db_path)
        
        print(f"Database file: {db_path.name}")
        print(f"Exists: {'Yes' if db_path.exists() else 'No'}")
        print(f"Validated documents: {count}")
        
        if db_path.exists():
            size_kb = db_path.stat().st_size / 1024
            print(f"Size: {size_kb:.1f} KB")
        
        print()
        
    except Exception as e:
        print(f"❌ Could not check database: {e}")
        print()

def show_architecture_details():
    """Show technical architecture details"""
    print("🏗️  System Architecture")
    print("-" * 40)
    
    print("Pipeline Flow:")
    print("  Input → Doctor Agent → Editor Agent → Moderator Agent → Database")
    print("             ↓              ↓              ↓")
    print("        Conflict Type   Modified Docs   Validation")
    print("                           ↑              ↓")
    print("                           └── Retry Loop (if invalid) ──┘")
    print()
    
    print("Key Components:")
    print("• Doctor Agent: Uses LLM to analyze document pairs")
    print("• Editor Agent: Modifies documents to create conflicts")
    print("• Moderator Agent: Validates modifications for realism")
    print("• Data Loader: Manages MIMIC-III clinical document dataset")
    print("• Database: SQLite storage for validated documents")
    print("• Pipeline Controller: Orchestrates the complete workflow")
    print()

def main():
    """Run the demo"""
    show_system_overview()
    show_conflict_types()
    show_dataset_info()
    show_sample_documents()
    show_database_status()
    show_api_requirements()
    show_usage_examples()
    show_architecture_details()
    
    print("🚀 Ready to start! Try running:")
    
    if os.getenv("GROQ_API_KEY"):
        print("   python main.py batch --size 2")
    else:
        print("   python main.py db-info")
        print("   (Set GROQ_API_KEY for full functionality)")

if __name__ == "__main__":
    main()
