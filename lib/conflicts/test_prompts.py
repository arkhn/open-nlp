#!/usr/bin/env python3
"""
Test script to verify the prompt loading system works correctly
"""

from prompt_loader import PromptLoader, load_prompt
from config import DOCTOR_AGENT_PROMPT_FILE, EDITOR_AGENT_PROMPT_FILE, MODERATOR_AGENT_PROMPT_FILE

def test_prompt_loader():
    """Test the PromptLoader functionality"""
    print("Testing PromptLoader...")
    
    # Test creating loader
    loader = PromptLoader()
    print(f"✓ Created PromptLoader with prompts directory: {loader.prompts_dir}")
    
    # Test listing available prompts
    available_prompts = loader.list_available_prompts()
    print(f"✓ Available prompts: {available_prompts}")
    
    # Test loading each agent prompt
    try:
        doctor_prompt = loader.load_prompt(DOCTOR_AGENT_PROMPT_FILE)
        print(f"✓ Loaded doctor prompt ({len(doctor_prompt)} chars)")
        
        editor_prompt = loader.load_prompt(EDITOR_AGENT_PROMPT_FILE)
        print(f"✓ Loaded editor prompt ({len(editor_prompt)} chars)")
        
        moderator_prompt = loader.load_prompt(MODERATOR_AGENT_PROMPT_FILE)
        print(f"✓ Loaded moderator prompt ({len(moderator_prompt)} chars)")
        
    except Exception as e:
        print(f"✗ Error loading prompts: {e}")
        return False
    
    # Test convenience function
    try:
        prompt_via_function = load_prompt(DOCTOR_AGENT_PROMPT_FILE)
        print(f"✓ Convenience function works ({len(prompt_via_function)} chars)")
    except Exception as e:
        print(f"✗ Error with convenience function: {e}")
        return False
    
    # Test prompt info
    try:
        info = loader.get_prompt_info(DOCTOR_AGENT_PROMPT_FILE)
        print(f"✓ Prompt info: {info}")
    except Exception as e:
        print(f"✗ Error getting prompt info: {e}")
        return False
    
    print("\n✅ All prompt loader tests passed!")
    return True

def test_prompt_formatting():
    """Test that prompts can be formatted correctly"""
    print("\nTesting prompt formatting...")
    
    try:
        # Test doctor prompt formatting
        doctor_prompt = load_prompt(DOCTOR_AGENT_PROMPT_FILE)
        formatted = doctor_prompt.format(
            conflict_types="Test conflict types",
            document1="Test document 1", 
            document2="Test document 2"
        )
        print(f"✓ Doctor prompt formats correctly ({len(formatted)} chars)")
        
        # Test editor prompt formatting  
        editor_prompt = load_prompt(EDITOR_AGENT_PROMPT_FILE)
        formatted = editor_prompt.format(
            conflict_type="Test Conflict",
            conflict_description="Test description",
            modification_instructions="Test instructions",
            document1="Test document 1",
            document2="Test document 2"
        )
        print(f"✓ Editor prompt formats correctly ({len(formatted)} chars)")
        
        # Test moderator prompt formatting
        moderator_prompt = load_prompt(MODERATOR_AGENT_PROMPT_FILE)
        formatted = moderator_prompt.format(
            original_doc1="Original doc 1",
            original_doc2="Original doc 2", 
            modified_doc1="Modified doc 1",
            modified_doc2="Modified doc 2",
            conflict_type="Test Conflict",
            changes_made="Test changes"
        )
        print(f"✓ Moderator prompt formats correctly ({len(formatted)} chars)")
        
    except Exception as e:
        print(f"✗ Error formatting prompts: {e}")
        return False
    
    print("✅ All prompt formatting tests passed!")
    return True

def show_prompt_samples():
    """Show sample content from each prompt"""
    print("\nPrompt Samples:")
    print("=" * 50)
    
    prompts = [
        (DOCTOR_AGENT_PROMPT_FILE, "Doctor Agent"),
        (EDITOR_AGENT_PROMPT_FILE, "Editor Agent"), 
        (MODERATOR_AGENT_PROMPT_FILE, "Moderator Agent")
    ]
    
    for prompt_file, name in prompts:
        try:
            content = load_prompt(prompt_file)
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"\n{name} Prompt Preview:")
            print("-" * 30)
            print(preview)
        except Exception as e:
            print(f"\n{name} Prompt Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Prompt Loading System")
    print("=" * 40)
    
    success = True
    success &= test_prompt_loader()
    success &= test_prompt_formatting()
    
    if success:
        show_prompt_samples()
        print("\n🎉 All tests completed successfully!")
        print("\n📝 Summary of improvements:")
        print("• Prompts are now stored in external .txt files")
        print("• No more large text blocks in config.py")
        print("• Easy to edit and version control prompts")
        print("• Caching system for better performance")
        print("• Error handling and validation")
    else:
        print("\n❌ Some tests failed - check the output above")
