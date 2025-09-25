#!/usr/bin/env python3
"""
BGE-M3 Test Runner Script

This script provides a convenient way to run BGE-M3 tests with different configurations.
It supports running specific test categories, generating coverage reports, and benchmarking.
"""

import subprocess
import sys
import argparse
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any


class BGE_M3_TestRunner:
    """Test runner for BGE-M3 tests"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.test_dir = self.base_dir / "tests"
        self.coverage_dir = self.base_dir / "htmlcov" / "bge_m3"
        self.coverage_dir.mkdir(parents=True, exist_ok=True)
        
    def run_command(self, command: List[str], cwd: Path = None) -> Dict[str, Any]:
        """Run a command and return the result"""
        if cwd is None:
            cwd = self.base_dir
            
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            end_time = time.time()
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "time": end_time - start_time,
                "command": " ".join(command)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
                "time": 300,
                "command": " ".join(command)
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "time": time.time() - start_time,
                "command": " ".join(command)
            }
    
    def run_all_tests(self, coverage: bool = True, verbose: bool = False) -> Dict[str, Any]:
        """Run all BGE-M3 tests"""
        print("Running all BGE-M3 tests...")
        
        command = ["pytest"]
        if coverage:
            command.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
        if verbose:
            command.append("-v")
        command.extend(["-m", "bge_m3"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ All BGE-M3 tests passed!")
        else:
            print("❌ Some BGE-M3 tests failed!")
            if result["stderr"]:
                print(f"Error: {result['stderr']}")
        
        return result
    
    def run_unit_tests(self, coverage: bool = True) -> Dict[str, Any]:
        """Run unit tests only"""
        print("Running BGE-M3 unit tests...")
        
        command = ["pytest", "-m", "bge_m3 and unit"]
        if coverage:
            command.extend(["--cov=src", "--cov-report=term-missing"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ All BGE-M3 unit tests passed!")
        else:
            print("❌ Some BGE-M3 unit tests failed!")
        
        return result
    
    def run_integration_tests(self, coverage: bool = True) -> Dict[str, Any]:
        """Run integration tests only"""
        print("Running BGE-M3 integration tests...")
        
        command = ["pytest", "-m", "bge_m3_integration"]
        if coverage:
            command.extend(["--cov=src", "--cov-report=term-missing"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ All BGE-M3 integration tests passed!")
        else:
            print("❌ Some BGE-M3 integration tests failed!")
        
        return result
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests only"""
        print("Running BGE-M3 performance tests...")
        
        command = ["pytest", "-m", "bge_m3_performance", "-v"]
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ All BGE-M3 performance tests passed!")
        else:
            print("❌ Some BGE-M3 performance tests failed!")
        
        return result
    
    def run_service_tests(self, coverage: bool = True) -> Dict[str, Any]:
        """Run BGE-M3 service tests only"""
        print("Running BGE-M3 service tests...")
        
        command = ["pytest", "tests/test_bge_m3_service.py"]
        if coverage:
            command.extend(["--cov=src/app/services/bge_m3_service.py", "--cov-report=term-missing"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ All BGE-M3 service tests passed!")
        else:
            print("❌ Some BGE-M3 service tests failed!")
        
        return result
    
    def run_utils_tests(self, coverage: bool = True) -> Dict[str, Any]:
        """Run BGE-M3 utils tests only"""
        print("Running BGE-M3 utils tests...")
        
        command = ["pytest", "tests/test_bge_m3_qdrant_utils.py"]
        if coverage:
            command.extend(["--cov=src/app/utils/qdrant_utils.py", "--cov-report=term-missing"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ All BGE-M3 utils tests passed!")
        else:
            print("❌ Some BGE-M3 utils tests failed!")
        
        return result
    
    def run_api_tests(self, coverage: bool = True) -> Dict[str, Any]:
        """Run BGE-M3 API tests only"""
        print("Running BGE-M3 API tests...")
        
        command = ["pytest", "tests/test_bge_m3_api.py"]
        if coverage:
            command.extend(["--cov=src/app/api/endpoints", "--cov-report=term-missing"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ All BGE-M3 API tests passed!")
        else:
            print("❌ Some BGE-M3 API tests failed!")
        
        return result
    
    def generate_coverage_report(self, format: str = "html") -> Dict[str, Any]:
        """Generate coverage report"""
        print(f"Generating {format} coverage report...")
        
        if format == "html":
            command = ["pytest", "--cov=src", "--cov-report=html"]
        elif format == "xml":
            command = ["pytest", "--cov=src", "--cov-report=xml"]
        elif format == "json":
            command = ["pytest", "--cov=src", "--cov-report=json"]
        else:
            print(f"Unsupported format: {format}")
            return {"success": False, "error": f"Unsupported format: {format}"}
        
        result = self.run_command(command)
        
        if result["success"]:
            print(f"✅ {format.upper()} coverage report generated!")
            if format == "html":
                print(f"📄 Report available at: {self.coverage_dir}/index.html")
        else:
            print(f"❌ Failed to generate {format} coverage report!")
        
        return result
    
    def run_benchmark(self, test_name: str = None, iterations: int = 10) -> Dict[str, Any]:
        """Run benchmark tests"""
        print(f"Running benchmark tests (iterations: {iterations})...")
        
        command = ["pytest", "-m", "benchmark", "-v"]
        if test_name:
            command.extend(["-k", test_name])
        command.extend(["--benchmark-only", f"--benchmark-sort=mean", f"--benchmark-min-rounds={iterations}"])
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ Benchmark tests completed!")
        else:
            print("❌ Benchmark tests failed!")
        
        return result
    
    def run_specific_test(self, test_path: str) -> Dict[str, Any]:
        """Run a specific test file or test function"""
        print(f"Running specific test: {test_path}")
        
        command = ["pytest", test_path, "-v"]
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
        
        return result
    
    def run_with_coverage_threshold(self, threshold: int = 90) -> Dict[str, Any]:
        """Run tests with coverage threshold"""
        print(f"Running tests with coverage threshold: {threshold}%")
        
        command = ["pytest", "--cov=src", f"--cov-fail-under={threshold}", "--cov-report=term-missing"]
        
        result = self.run_command(command)
        
        if result["success"]:
            print(f"✅ Tests passed with {threshold}%+ coverage!")
        else:
            print(f"❌ Tests failed to meet {threshold}% coverage threshold!")
        
        return result
    
    def run_parallel_tests(self, processes: int = None) -> Dict[str, Any]:
        """Run tests in parallel"""
        if processes is None:
            processes = os.cpu_count() or 4
        
        print(f"Running tests in parallel with {processes} processes...")
        
        command = ["pytest", "-n", str(processes), "-m", "bge_m3"]
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ Parallel tests completed!")
        else:
            print("❌ Parallel tests failed!")
        
        return result
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        print("Generating comprehensive test report...")
        
        # Run all test categories
        results = {
            "unit_tests": self.run_unit_tests(coverage=False),
            "integration_tests": self.run_integration_tests(coverage=False),
            "service_tests": self.run_service_tests(coverage=False),
            "utils_tests": self.run_utils_tests(coverage=False),
            "api_tests": self.run_api_tests(coverage=False),
            "performance_tests": self.run_performance_tests(),
        }
        
        # Generate coverage report
        coverage_result = self.generate_coverage_report("html")
        
        # Calculate overall success
        all_passed = all(result["success"] for result in results.values())
        
        # Generate summary
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_success": all_passed,
            "test_results": results,
            "coverage_success": coverage_result["success"],
            "total_tests": len(results),
            "passed_tests": sum(1 for result in results.values() if result["success"]),
            "failed_tests": sum(1 for result in results.values() if not result["success"]),
        }
        
        # Save report
        report_file = self.coverage_dir / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Test report saved to: {report_file}")
        
        if all_passed and coverage_result["success"]:
            print("✅ All tests passed and coverage generated!")
        else:
            print("❌ Some tests failed or coverage generation failed!")
        
        return summary
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run health check tests"""
        print("Running BGE-M3 health checks...")
        
        command = ["pytest", "-m", "bge_m3", "--tb=short"]
        
        result = self.run_command(command)
        
        if result["success"]:
            print("✅ BGE-M3 health checks passed!")
        else:
            print("❌ BGE-M3 health checks failed!")
        
        return result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="BGE-M3 Test Runner")
    parser.add_argument("--mode", choices=[
        "all", "unit", "integration", "service", "utils", "api", "performance", "benchmark", "health"
    ], default="all", help="Test mode to run")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage reports")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--threshold", type=int, default=90, help="Coverage threshold percentage")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--processes", type=int, help="Number of parallel processes")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--benchmark-iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--test-path", help="Specific test path to run")
    parser.add_argument("--coverage-format", choices=["html", "xml", "json"], default="html", help="Coverage report format")
    
    args = parser.parse_args()
    
    runner = BGE_M3_TestRunner()
    
    try:
        if args.mode == "all":
            result = runner.run_all_tests(coverage=args.coverage, verbose=args.verbose)
        elif args.mode == "unit":
            result = runner.run_unit_tests(coverage=args.coverage)
        elif args.mode == "integration":
            result = runner.run_integration_tests(coverage=args.coverage)
        elif args.mode == "service":
            result = runner.run_service_tests(coverage=args.coverage)
        elif args.mode == "utils":
            result = runner.run_utils_tests(coverage=args.coverage)
        elif args.mode == "api":
            result = runner.run_api_tests(coverage=args.coverage)
        elif args.mode == "performance":
            result = runner.run_performance_tests()
        elif args.mode == "benchmark":
            result = runner.run_benchmark(iterations=args.benchmark_iterations)
        elif args.mode == "health":
            result = runner.run_health_check()
        
        if args.coverage:
            runner.generate_coverage_report(args.coverage_format)
        
        if args.threshold:
            runner.run_with_coverage_threshold(args.threshold)
        
        if args.parallel:
            runner.run_parallel_tests(args.processes)
        
        if args.report:
            runner.generate_test_report()
        
        if args.test_path:
            runner.run_specific_test(args.test_path)
        
        # Exit with appropriate code
        sys.exit(0 if result["success"] else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()