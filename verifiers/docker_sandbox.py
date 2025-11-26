"""
Docker-based sandboxed code execution.

Provides secure, isolated environment for running untrusted code with:
- Resource limits (CPU, memory, timeout)
- No network access
- Read-only filesystem (except /tmp)
- Container pooling for performance
"""

import time
import docker
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from threading import Lock
from queue import Queue, Empty

from .base import VerificationError, TimeoutError


logger = logging.getLogger(__name__)


# ============================================================================
# Execution Result
# ============================================================================

@dataclass
class ExecutionResult:
    """Result of code execution in Docker container."""
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    timed_out: bool = False
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """Check if execution completed successfully."""
        return self.exit_code == 0 and not self.timed_out and self.error is None


# ============================================================================
# Docker Sandbox Manager
# ============================================================================

class DockerSandbox:
    """
    Manages Docker containers for sandboxed code execution.

    Features:
    - Container pooling for <100ms startup time
    - Resource limits (CPU, memory, timeout)
    - Security: no network, read-only filesystem
    - Automatic cleanup of idle containers
    - Multi-language support (Python, JavaScript, Bash, SQL)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Docker sandbox manager.

        Args:
            config: Configuration dictionary with keys:
                - docker_image: Docker image to use
                - container_pool_size: Number of warm containers (default: 5)
                - memory_limit: Memory limit string (e.g., "512m")
                - cpu_limit: CPU limit as float (e.g., 2.0 for 2 cores)
                - timeout: Execution timeout in seconds
                - network_disabled: Whether to disable network (default: True)

        Raises:
            VerificationError: If Docker is not available or image doesn't exist
        """
        self.config = config
        self.image = config.get("docker_image", "nexus-code-verifier:latest")
        self.pool_size = config.get("container_pool_size", 5)
        self.memory_limit = config.get("memory_limit", "512m")
        self.cpu_limit = config.get("cpu_limit", 2.0)
        self.timeout = config.get("timeout", 5.0)
        self.network_disabled = config.get("network_disabled", True)
        self.max_output_size = config.get("max_output_size", 10000)

        # Initialize Docker client
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            raise VerificationError(f"Failed to connect to Docker: {e}")

        # Verify image exists
        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            logger.warning(f"Docker image '{self.image}' not found. Container pool disabled.")
            self.pool_size = 0  # Disable pooling if image doesn't exist

        # Container pool (queue of ready-to-use containers)
        self._pool: Queue = Queue(maxsize=self.pool_size)
        self._pool_lock = Lock()
        self._active_containers: List[Any] = []

        # Initialize container pool
        if self.pool_size > 0:
            self._initialize_pool()

    def _initialize_pool(self):
        """Pre-create containers for the pool."""
        logger.info(f"Initializing container pool with {self.pool_size} containers...")
        for i in range(self.pool_size):
            try:
                container = self._create_container()
                self._pool.put(container)
                logger.debug(f"Added container {i+1}/{self.pool_size} to pool")
            except Exception as e:
                logger.error(f"Failed to create pooled container: {e}")

    def _create_container(self) -> Any:
        """
        Create a new Docker container with security and resource limits.

        Returns:
            Docker container object

        Raises:
            VerificationError: If container creation fails
        """
        try:
            # Security settings
            network_mode = "none" if self.network_disabled else "bridge"

            container = self.client.containers.create(
                image=self.image,
                command="tail -f /dev/null",  # Keep container alive
                detach=True,
                mem_limit=self.memory_limit,
                nano_cpus=int(self.cpu_limit * 1e9),  # Convert to nanocpus
                network_mode=network_mode,
                read_only=True,  # Read-only root filesystem
                tmpfs={'/tmp': 'rw,noexec,nosuid,size=100m'},  # Writable /tmp, no execution
                security_opt=['no-new-privileges'],  # Prevent privilege escalation
                cap_drop=['ALL'],  # Drop all capabilities
                user='1000:1000',  # Non-root user
            )

            container.start()
            return container

        except Exception as e:
            raise VerificationError(f"Failed to create container: {e}")

    def _get_container(self) -> Any:
        """
        Get a container from the pool or create a new one.

        Returns:
            Docker container object
        """
        # Try to get from pool (non-blocking)
        try:
            container = self._pool.get_nowait()
            # Verify container is still running
            container.reload()
            if container.status != 'running':
                container.remove(force=True)
                raise Empty  # Will create new container below
            return container
        except Empty:
            # Pool empty or container invalid - create new one
            return self._create_container()

    def _return_container(self, container: Any):
        """
        Return a container to the pool or remove it.

        Args:
            container: Docker container object
        """
        try:
            # Check if container is still healthy
            container.reload()
            if container.status == 'running' and not self._pool.full():
                # Return to pool
                self._pool.put_nowait(container)
            else:
                # Pool full or container unhealthy - remove it
                container.remove(force=True)
                # Trigger replenishment if pool is below target
                self._maybe_replenish_pool()
        except Exception as e:
            logger.warning(f"Error returning container to pool: {e}")
            try:
                container.remove(force=True)
            except Exception:
                pass
            # Trigger replenishment after container removal
            self._maybe_replenish_pool()

    def _maybe_replenish_pool(self):
        """
        Replenish the container pool if below target size.

        Called after containers are removed to maintain pool health.
        """
        if self.pool_size == 0:
            return  # Pooling disabled

        with self._pool_lock:
            current_size = self._pool.qsize()
            needed = self.pool_size - current_size

            if needed > 0:
                logger.debug(f"Pool replenishment: {current_size}/{self.pool_size} containers, adding {needed}")
                for _ in range(needed):
                    try:
                        container = self._create_container()
                        if not self._pool.full():
                            self._pool.put_nowait(container)
                        else:
                            # Pool filled by another thread, remove extra
                            container.remove(force=True)
                            break
                    except Exception as e:
                        logger.warning(f"Failed to replenish container pool: {e}")
                        break  # Stop trying if creation fails

    def execute(self, code: str, language: str, stdin: str = "",
                env: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """
        Execute code in a sandboxed Docker container.

        Args:
            code: Code to execute
            language: Programming language ('python', 'javascript', 'bash', 'sql')
            stdin: Standard input to provide to the program
            env: Optional environment variables

        Returns:
            ExecutionResult with stdout, stderr, exit code

        Raises:
            VerificationError: If execution fails
            TimeoutError: If execution exceeds timeout
        """
        start_time = time.time()
        container = None

        try:
            # Get container from pool
            container = self._get_container()

            # Prepare execution command based on language
            cmd = self._get_exec_command(code, language)

            # Execute with timeout
            exec_result = container.exec_run(
                cmd,
                stdin=True,
                stdout=True,
                stderr=True,
                demux=True,
                environment=env or {},
            )

            execution_time = time.time() - start_time

            # Check for timeout (exit code 124 from timeout command, or wall time exceeded)
            # Exit code 124 = command timed out
            # Exit code 137 = killed by SIGKILL (9+128)
            timed_out = (
                exec_result.exit_code == 124 or
                exec_result.exit_code == 137 or
                execution_time >= self.timeout
            )

            # Decode output (exec_result.output is tuple of (stdout, stderr))
            stdout_bytes, stderr_bytes = exec_result.output if exec_result.output else (b'', b'')
            stdout = (stdout_bytes or b'').decode('utf-8', errors='replace')
            stderr = (stderr_bytes or b'').decode('utf-8', errors='replace')

            # Truncate output if too large
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + f"\n... (truncated, {len(stdout)} total chars)"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + f"\n... (truncated, {len(stderr)} total chars)"

            return ExecutionResult(
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                exit_code=exec_result.exit_code,
                execution_time=execution_time,
                timed_out=timed_out,
            )

        except docker.errors.APIError as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=time.time() - start_time,
                error=f"Docker API error: {e}",
            )

        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=time.time() - start_time,
                error=f"Execution error: {e}",
            )

        finally:
            # Return container to pool
            if container:
                self._return_container(container)

    def _get_exec_command(self, code: str, language: str) -> List[str]:
        """
        Generate execution command for the given language.

        Wraps command with `timeout` to enforce execution time limit.

        Args:
            code: Code to execute
            language: Programming language

        Returns:
            List of command arguments

        Raises:
            VerificationError: If language is not supported
        """
        language = language.lower()

        # Base command based on language
        if language == 'python':
            base_cmd = ['python3', '-c', code]

        elif language in ['javascript', 'js', 'node']:
            base_cmd = ['node', '-e', code]

        elif language in ['bash', 'sh', 'shell']:
            base_cmd = ['bash', '-c', code]

        elif language == 'sql':
            # Use SQLite in-memory database
            base_cmd = ['sqlite3', ':memory:', code]

        else:
            raise VerificationError(f"Unsupported language: {language}")

        # Wrap with timeout command for enforcement
        # timeout sends SIGTERM, then SIGKILL after grace period (-k flag)
        # This ensures long-running code is actually terminated
        timeout_seconds = int(self.timeout) + 1  # Add 1 second grace
        return ['timeout', '-k', '2', str(timeout_seconds)] + base_cmd

    def execute_with_tests(self, code: str, language: str,
                           test_cases: List[Dict[str, Any]]) -> List[ExecutionResult]:
        """
        Execute code with multiple test cases.

        Args:
            code: Code to execute (usually a function definition)
            language: Programming language
            test_cases: List of test case dicts with keys:
                - input: Input to provide
                - expected_output: Expected output
                - description: Test case description (optional)

        Returns:
            List of ExecutionResult objects, one per test case

        Example test_cases:
            [
                {"input": "5", "expected_output": "120"},
                {"input": "3", "expected_output": "6"},
            ]
        """
        results = []

        for i, test_case in enumerate(test_cases):
            stdin = test_case.get('input', '')
            expected = test_case.get('expected_output')

            # Execute code with this test case's input
            result = self.execute(code, language, stdin=stdin)

            # Add test case metadata
            result.test_case_index = i
            result.expected_output = expected
            result.description = test_case.get('description', f"Test case {i+1}")

            results.append(result)

        return results

    def cleanup(self):
        """Clean up all containers in the pool and active containers."""
        logger.info("Cleaning up Docker sandbox...")

        # Clean up pooled containers
        while not self._pool.empty():
            try:
                container = self._pool.get_nowait()
                container.remove(force=True)
                logger.debug("Removed pooled container")
            except Exception as e:
                logger.warning(f"Error removing pooled container: {e}")

        # Clean up active containers
        for container in self._active_containers:
            try:
                container.remove(force=True)
                logger.debug("Removed active container")
            except Exception as e:
                logger.warning(f"Error removing active container: {e}")

        self._active_containers.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup containers."""
        self.cleanup()

    def __del__(self):
        """Destructor - cleanup containers."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor


# ============================================================================
# Convenience Functions
# ============================================================================

def execute_code(code: str, language: str, config: Optional[Dict[str, Any]] = None,
                 stdin: str = "") -> ExecutionResult:
    """
    Convenience function to execute code in a Docker sandbox.

    Args:
        code: Code to execute
        language: Programming language
        config: Optional configuration dictionary
        stdin: Optional standard input

    Returns:
        ExecutionResult

    Example:
        >>> result = execute_code("print(2 + 2)", "python")
        >>> print(result.stdout)
        4
    """
    config = config or {}
    with DockerSandbox(config) as sandbox:
        return sandbox.execute(code, language, stdin=stdin)
