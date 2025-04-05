import asyncio
import unittest
from unittest import mock
from graphrag.utils import run_async_function  # Replace 'your_module' with the actual module name


async def sample_async_function(x, y):
    """A simple async function for testing."""
    return x + y


class TestRunAsyncFunction(unittest.TestCase):

    @mock.patch("asyncio.get_running_loop")
    def test_run_without_event_loop(self, mock_get_running_loop):
        # Simulate no event loop by raising RuntimeError
        mock_get_running_loop.side_effect = RuntimeError

        # Call run_async_function with a sample async function
        result = run_async_function(sample_async_function, 2, 3)

        # Assert that the result is correct
        self.assertEqual(result, 5)

    @mock.patch("asyncio.get_running_loop")
    @mock.patch("nest_asyncio.apply")  # Mock nest_asyncio to avoid side effects
    def test_run_with_event_loop_in_notebook(self, mock_nest_apply, mock_get_running_loop):
        # Simulate an active event loop
        mock_loop = mock.Mock()
        mock_get_running_loop.return_value = mock_loop
        mock_loop.is_running.return_value = True

        # Use a mock to replace loop.run_until_complete
        with mock.patch.object(mock_loop, "run_until_complete", return_value=42) as mock_run_until_complete:
            # Call run_async_function with a sample async function
            result = run_async_function(sample_async_function, 6, 7)

            # Assert that nest_asyncio.apply was called
            mock_nest_apply.assert_called_once()

            # Verify that run_until_complete was called with the proper coroutine
            mock_run_until_complete.assert_called_once()
            coroutine_passed = mock_run_until_complete.call_args[0][0]  # Get the coroutine passed
            self.assertEqual(coroutine_passed.__name__, sample_async_function.__name__)  # Function name should match
            self.assertEqual(coroutine_passed.cr_frame.f_locals['x'], 6)  # Check argument `x`
            self.assertEqual(coroutine_passed.cr_frame.f_locals['y'], 7)  # Check argument `y`

            # Assert the correct result
            self.assertEqual(result, 42)


