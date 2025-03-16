import os
import unittest

if __name__ == "__main__":
    # Discover all integration test cases in `tests/integration/` directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=current_dir, pattern="test*.py")

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit code based on test results
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)
