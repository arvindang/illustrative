import sys
try:
    import scripting_agent
    import illustrator_agent
    print("✅ Syntax check passed: Modules imported successfully.")
except Exception as e:
    print(f"❌ Syntax check failed: {e}")
    sys.exit(1)
