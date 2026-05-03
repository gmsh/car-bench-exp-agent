"""Public surface of the purple agent package.

The executor is exposed at the package root so that the legacy
``car_bench_agent`` shim and ``server.py`` can import it as
``from car_bench_agent import CARBenchAgentExecutor`` /
``from agent import CARBenchAgentExecutor`` without knowing the
internal layout.
"""
from .executor import CARBenchAgentExecutor, logger

__all__ = ["CARBenchAgentExecutor", "logger"]
