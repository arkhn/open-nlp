#!/usr/bin/env python3
"""
Test script for the Clinical Document Conflict Pipeline
"""

import os
import sys

def test_imports():
    """Test that all modules can be imported successfully"""
    print("Testing imports...")
    
    try:
        from config import CONFLICT_TYPES, GROQ_MODEL
        print("✓ Config imported successfully")
        
        from base import DocumentPair, ConflictResult, EditorResult, ValidationResult
        print("✓ Base classes imported successfully")
        
        from data_loader import ClinicalDataLoader, create_sample_data_if_missing
        print("✓ Data loader imported successfully")
        
        from agents.doctor_agent import DoctorAgent
        from agents.editor_agent import EditorAgent
        from agents.moderator_agent import ModeratorAgent
        print("✓ All agents imported successfully")
        
        from pipeline import ClinicalConflictPipeline
        print("✓ Pipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration and conflict types"""
    print("\nTesting configuration...")
    
    try:
        from config import CONFLICT_TYPES
        print(f"✓ Found {len(CONFLICT_TYPES)} conflict types:")
        for key, conflict_type in CONFLICT_TYPES.items():
            print(f"  - {key}: {conflict_type.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation"""
    print("\nTesting sample data creation...")
    
    try:
        from data_loader import create_sample_data_if_missing
        
        # This will create sample data if it doesn't exist
        was_created = create_sample_data_if_missing()
        
        if was_created:
            print("✓ Sample data created successfully")
        else:
            print("✓ Real dataset already exists, no sample data needed")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation failed: {e}")
        return False

def test_api_key_setup():
    """Test Groq API key setup"""
    print("\nTesting API key setup...")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if groq_api_key:
        print("✓ GROQ_API_KEY environment variable is set")
        return True
    else:
        print("✗ GROQ_API_KEY environment variable is not set")
        print("  Please set it with: export GROQ_API_KEY='your_api_key_here'")
        return False

def test_data_loader():
    """Test data loader functionality"""
    print("\nTesting data loader...")
    
    try:
        from data_loader import ClinicalDataLoader
        
        loader = ClinicalDataLoader()
        stats = loader.get_data_statistics()
        
        print(f"✓ Loaded {stats['total_documents']} documents")
        print(f"✓ Found {stats['unique_subjects']} unique subjects")
        print(f"✓ Available categories: {', '.join(stats['sample_categories'])}")
        
        # Test document pair generation
        pairs = loader.get_random_document_pairs(count=1)
        print(f"✓ Successfully generated {len(pairs)} document pair(s)")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_agents_instantiation():
    """Test that all agents can be instantiated"""
    print("\nTesting agent instantiation...")
    
    # Skip this test if no API key is available
    if not os.getenv("GROQ_API_KEY"):
        print("⚠ Skipping agent test (no API key)")
        return True
    
    try:
        from agents.doctor_agent import DoctorAgent
        from agents.editor_agent import EditorAgent
        from agents.moderator_agent import ModeratorAgent
        
        doctor = DoctorAgent()
        editor = EditorAgent()
        moderator = ModeratorAgent()
        
        print("✓ Doctor Agent instantiated")
        print("✓ Editor Agent instantiated") 
        print("✓ Moderator Agent instantiated")
        
        # Test doctor agent conflict type listing
        conflict_types = doctor.list_all_conflict_types()
        print(f"✓ Doctor Agent lists {len(conflict_types)} conflict types")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent instantiation failed: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation"""
    print("\nTesting pipeline creation...")
    
    # Skip this test if no API key is available
    if not os.getenv("GROQ_API_KEY"):
        print("⚠ Skipping pipeline test (no API key)")
        return True
    
    try:
        from pipeline import ClinicalConflictPipeline
        
        pipeline = ClinicalConflictPipeline()
        stats = pipeline.get_pipeline_statistics()
        
        print("✓ Pipeline created successfully")
        print(f"✓ Pipeline has {stats['validated_documents']} validated documents in database")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Clinical Document Conflict Pipeline - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_sample_data_creation,
        test_api_key_setup,
        test_data_loader,
        test_agents_instantiation,
        test_pipeline_creation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Your pipeline is ready to use.")
        print("\nNext steps:")
        if not os.getenv("GROQ_API_KEY"):
            print("1. Set your GROQ_API_KEY environment variable")
        print("2. Run: python main.py batch --size 2")
        print("3. Or: python main.py list-conflicts")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
        if not os.getenv("GROQ_API_KEY"):
            print("\n💡 Tip: Most functionality requires the GROQ_API_KEY to be set")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
