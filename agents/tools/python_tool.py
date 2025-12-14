"""
Python scratchpad tool for code execution
Provides a safe environment for agents to run Python code
"""

import sys
import io
import contextlib
from typing import Dict, Any, Optional
import traceback


class PythonScratchpad:
    """
    Safe Python execution environment for agents.
    Allows agents to write and execute Python code to solve problems.
    """

    def __init__(self, timeout: int = 10, max_output_length: int = 5000):
        """
        Initialize Python scratchpad

        Args:
            timeout: Maximum execution time in seconds (not strictly enforced)
            max_output_length: Maximum length of output to return
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.variables = {}  # Persistent variable storage across executions

    def execute(self, code: str, reset_variables: bool = False) -> str:
        """
        Execute Python code in a controlled environment

        Args:
            code: Python code to execute
            reset_variables: Whether to reset stored variables

        Returns:
            Output from code execution (stdout + result)
        """
        if reset_variables:
            self.variables = {}

        # Create safe execution environment
        safe_globals = {
            '__builtins__': {
                # Math and basic operations
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'reversed': reversed,
                'pow': pow,
                'divmod': divmod,

                # Data structures
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,

                # Utilities
                'print': print,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'any': any,
                'all': all,

                # Common exceptions
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
            }
        }

        # Add safe mathematical libraries
        try:
            import math
            import statistics
            import itertools
            import collections
            import json
            import re

            safe_globals['math'] = math
            safe_globals['statistics'] = statistics
            safe_globals['itertools'] = itertools
            safe_globals['collections'] = collections
            safe_globals['json'] = json
            safe_globals['re'] = re
        except ImportError:
            pass

        # Merge with persistent variables
        safe_globals.update(self.variables)

        # Capture stdout
        output_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(output_buffer):
                # Execute the code
                exec_result = None

                # Try to eval first (for expressions)
                try:
                    exec_result = eval(code, safe_globals)
                except SyntaxError:
                    # If eval fails, use exec (for statements)
                    exec(code, safe_globals)

                # Update persistent variables
                # Remove built-in modules and functions
                for key, value in safe_globals.items():
                    if key not in ['__builtins__', 'math', 'statistics',
                                   'itertools', 'collections', 'json', 're']:
                        self.variables[key] = value

                # Get output
                stdout_output = output_buffer.getvalue()

                # Combine stdout and result
                output_parts = []
                if stdout_output:
                    output_parts.append(stdout_output.strip())
                if exec_result is not None:
                    output_parts.append(f"Result: {exec_result}")

                final_output = "\n".join(output_parts) if output_parts else "Code executed successfully (no output)"

                # Truncate if too long
                if len(final_output) > self.max_output_length:
                    final_output = final_output[:self.max_output_length] + f"\n... (truncated, {len(final_output)} total chars)"

                return final_output

        except Exception as e:
            # Return error with traceback
            error_trace = traceback.format_exc()
            return f"Execution Error:\n{error_trace}"

    def reset(self):
        """Clear all stored variables"""
        self.variables = {}

    def get_variables(self) -> Dict[str, Any]:
        """Get all stored variables"""
        return self.variables.copy()

    def __call__(self, code: str) -> str:
        """Allow tool to be called directly"""
        return self.execute(code)


# Example usage demonstrations
if __name__ == "__main__":
    # Create scratchpad
    scratchpad = PythonScratchpad()

    # Test 1: Simple calculation
    print("Test 1: Simple calculation")
    result = scratchpad.execute("2 + 2")
    print(result)
    print()

    # Test 2: Variable persistence
    print("Test 2: Variable persistence")
    result1 = scratchpad.execute("x = 10\ny = 20\nprint(f'x={x}, y={y}')")
    print(result1)
    result2 = scratchpad.execute("z = x + y\nprint(f'z={z}')")
    print(result2)
    print()

    # Test 3: Complex calculation
    print("Test 3: Compound interest calculation")
    code = """
import math

# Compound interest formula
P = 10000  # Principal
r = 0.045  # Annual rate
n = 12     # Compounds per year
t = 7      # Years

A = P * (1 + r/n) ** (n * t)
print(f'Principal: ${P}')
print(f'Rate: {r*100}%')
print(f'Time: {t} years')
print(f'Final Amount: ${A:.2f}')
print(f'Interest Earned: ${A - P:.2f}')

A  # Return final amount
"""
    result = scratchpad.execute(code)
    print(result)
    print()

    # Test 4: Error handling
    print("Test 4: Error handling")
    result = scratchpad.execute("1 / 0")
    print(result)
    print()

    # Test 5: Solving a logic puzzle programmatically
    print("Test 5: Logic puzzle solving")
    code = """
# Solve: If Alice is older than Bob, and Bob is older than Carol,
# and Alice is 25, and Carol is 20, how old is Bob?

alice_age = 25
carol_age = 20

# Bob must be between Carol and Alice
possible_ages = range(carol_age + 1, alice_age)
bob_age = list(possible_ages)[len(possible_ages) // 2]  # Middle value

print(f"Alice: {alice_age}")
print(f"Carol: {carol_age}")
print(f"Bob (estimated): {bob_age}")
"""
    result = scratchpad.execute(code)
    print(result)
