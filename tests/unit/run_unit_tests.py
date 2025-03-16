import unittest

if __name__ == "__main__":
    # Discover all unit test cases in `tests/unit/` directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=".", pattern="test*.py")

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit code based on test results
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)
