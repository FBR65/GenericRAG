#!/usr/bin/env python3
"""
Test runner script for the Generic RAG System.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run the test suite."""
    print("Running Generic RAG System Tests")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "--color=yes",
        "--disable-warnings"
    ]
    
    try:
        # Run the tests
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\nAll tests passed!")
            print("\nCoverage report generated in htmlcov/index.html")
            return True
        else:
            print(f"\nTests failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\nError running tests: {str(e)}")
        return False

def run_specific_test(test_name):
    """Run a specific test file or test function."""
    print(f"Running specific test: {test_name}")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    cmd = [
        sys.executable, "-m", "pytest",
        test_name,
        "-v",
        "--tb=short",
        "--color=yes",
        "--disable-warnings"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running test: {str(e)}")
        return False

def run_with_coverage():
    """Run tests with coverage report."""
    print("Running Tests with Coverage Report")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--color=yes",
        "--disable-warnings"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests with coverage: {str(e)}")
        return False

def lint_code():
    """Run code linting."""
    print("Running Code Linting")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run ruff
    print("Running ruff...")
    cmd_ruff = [sys.executable, "-m", "ruff", "src/", "tests/"]
    result_ruff = subprocess.run(cmd_ruff, capture_output=True, text=True)
    
    if result_ruff.returncode != 0:
        print("Ruff found issues:")
        print(result_ruff.stdout)
        print(result_ruff.stderr)
        return False
    else:
        print("Ruff passed!")
    
    # Run black (if available)
    try:
        print("Running black...")
        cmd_black = [sys.executable, "-m", "black", "--check", "src/", "tests/"]
        result_black = subprocess.run(cmd_black, capture_output=True, text=True)
        
        if result_black.returncode != 0:
            print("Black formatting issues found")
            print("Run: black src/ tests/")
            return False
        else:
            print("Black formatting passed!")
    except FileNotFoundError:
        print("Black not installed, skipping...")
    
    return True

def main():
    """Main function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            run_tests()
        elif command == "coverage":
            run_with_coverage()
        elif command == "lint":
            lint_code()
        elif command == "specific":
            if len(sys.argv) > 2:
                run_specific_test(sys.argv[2])
            else:
                print("Please specify a test file or function")
                sys.exit(1)
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        # Default: run tests
        success = run_tests()
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()