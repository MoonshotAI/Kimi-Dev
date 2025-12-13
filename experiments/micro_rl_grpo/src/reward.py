"""
Reward Computation for GRPO Training

Implements test-based reward signal following Kimi-Dev paradigm:
- Fail-to-Pass (F2P): Tests that should pass after fix
- Pass-to-Pass (P2P): Tests that should remain passing
"""

import logging
import tempfile
import shutil
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a test execution."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    tests_passed: int
    tests_failed: int
    tests_total: int
    f2p_score: float  # Fail-to-Pass score
    p2p_score: float  # Pass-to-Pass score


@dataclass
class RewardResult:
    """Result of reward computation."""

    reward: float
    resolved: bool
    f2p_score: float
    p2p_score: float
    patch_applied: bool
    patch_valid: bool
    details: Dict[str, Any]


class PatchValidator:
    """Validates generated patches."""

    @staticmethod
    def validate_patch_format(patch: str) -> Tuple[bool, str]:
        """
        Validate patch is in correct unified diff format.

        Returns:
            (is_valid, error_message)
        """

        if not patch or not patch.strip():
            return False, "Empty patch"

        # Check for diff markers
        if "@@" not in patch:
            return False, "Missing hunk markers (@@)"

        # Check for file headers
        if not ("---" in patch and "+++" in patch):
            return False, "Missing file headers (--- / +++)"

        # Check for actual changes
        lines = patch.split("\n")
        has_addition = any(line.startswith("+") and not line.startswith("+++") for line in lines)
        has_deletion = any(line.startswith("-") and not line.startswith("---") for line in lines)

        if not (has_addition or has_deletion):
            return False, "No actual changes in patch"

        return True, ""

    @staticmethod
    def validate_search_replace(text: str) -> Tuple[bool, List[Dict[str, str]]]:
        """
        Validate and parse SEARCH/REPLACE format.

        Returns:
            (is_valid, list_of_edits)
        """

        edits = []

        # Pattern for SEARCH/REPLACE blocks
        pattern = r'### ([^\n]+)\n<{4,} SEARCH\n(.*?)\n={4,}\n(.*?)\n>{4,} REPLACE'

        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return False, []

        for match in matches:
            file_name, search_str, replace_str = match
            edits.append({
                "file": file_name.strip(),
                "search": search_str,
                "replace": replace_str,
            })

        return True, edits

    @staticmethod
    def validate_python_syntax(code: str) -> bool:
        """Check if code is valid Python syntax."""
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class TestExecutor:
    """Executes tests in sandboxed environment."""

    def __init__(
        self,
        timeout: int = 300,
        max_workers: int = 10,
    ):
        self.timeout = timeout
        self.max_workers = max_workers

    def execute_tests(
        self,
        repo_path: str,
        test_command: str,
        env: Optional[Dict[str, str]] = None,
    ) -> TestResult:
        """
        Execute tests and return results.

        Args:
            repo_path: Path to repository
            test_command: Command to run tests
            env: Environment variables

        Returns:
            TestResult with execution details
        """

        start_time = time.time()

        try:
            # Run tests
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, **(env or {})},
            )

            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode

            # Parse test results
            tests_passed, tests_failed, tests_total = self._parse_test_output(stdout + stderr)

            duration = time.time() - start_time

            return TestResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_total=tests_total,
                f2p_score=0.0,  # Computed later
                p2p_score=0.0,  # Computed later
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Test execution timed out",
                duration=self.timeout,
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
                f2p_score=0.0,
                p2p_score=0.0,
            )

        except Exception as e:
            return TestResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=time.time() - start_time,
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
                f2p_score=0.0,
                p2p_score=0.0,
            )

    def _parse_test_output(self, output: str) -> Tuple[int, int, int]:
        """Parse pytest/unittest output to get test counts."""

        # Try pytest format: "X passed, Y failed"
        pytest_pattern = r'(\d+) passed'
        pytest_failed = r'(\d+) failed'

        passed_match = re.search(pytest_pattern, output)
        failed_match = re.search(pytest_failed, output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0

        return passed, failed, passed + failed


class RewardComputer:
    """
    Computes rewards for GRPO training.

    Reward signal is based on test execution:
    - Full resolution (F2P=1, P2P=1): reward = 1.0
    - Partial resolution: reward = 0.5 * F2P + 0.5 * P2P
    - No resolution: reward = 0.0 or negative
    """

    def __init__(
        self,
        sandbox_manager=None,
        test_executor: Optional[TestExecutor] = None,
        timeout: int = 300,
        reward_scale: float = 1.0,
        penalty_invalid_patch: float = -0.5,
        penalty_failed_apply: float = -0.3,
    ):
        self.sandbox_manager = sandbox_manager
        self.test_executor = test_executor or TestExecutor(timeout=timeout)
        self.timeout = timeout
        self.reward_scale = reward_scale
        self.penalty_invalid_patch = penalty_invalid_patch
        self.penalty_failed_apply = penalty_failed_apply

        self.validator = PatchValidator()

    def compute_reward(
        self,
        problem_data: Dict[str, Any],
        completion: str,
    ) -> float:
        """
        Compute reward for a completion.

        Args:
            problem_data: Problem data from SWE-bench
            completion: Model's completion (code edit)

        Returns:
            Reward value
        """

        result = self.evaluate_completion(problem_data, completion)
        return result.reward

    def evaluate_completion(
        self,
        problem_data: Dict[str, Any],
        completion: str,
    ) -> RewardResult:
        """
        Full evaluation of a completion.

        Args:
            problem_data: Problem data including:
                - instance_id
                - repo_path or repo structure
                - FAIL_TO_PASS tests
                - PASS_TO_PASS tests
                - test_command
            completion: Model's completion

        Returns:
            RewardResult with detailed metrics
        """

        details = {
            "instance_id": problem_data.get("instance_id", "unknown"),
            "completion_length": len(completion),
        }

        # Step 1: Parse completion to get patch
        patch, patch_valid, parse_error = self._parse_completion(completion)
        details["patch_valid"] = patch_valid
        details["parse_error"] = parse_error

        if not patch_valid:
            return RewardResult(
                reward=self.penalty_invalid_patch * self.reward_scale,
                resolved=False,
                f2p_score=0.0,
                p2p_score=0.0,
                patch_applied=False,
                patch_valid=False,
                details=details,
            )

        # Step 2: Apply patch and run tests
        test_result = self._run_tests_with_patch(
            problem_data=problem_data,
            patch=patch,
        )

        details["test_result"] = {
            "exit_code": test_result.exit_code,
            "tests_passed": test_result.tests_passed,
            "tests_failed": test_result.tests_failed,
            "duration": test_result.duration,
        }

        if not test_result.success and test_result.exit_code == -2:
            # Patch failed to apply
            return RewardResult(
                reward=self.penalty_failed_apply * self.reward_scale,
                resolved=False,
                f2p_score=0.0,
                p2p_score=0.0,
                patch_applied=False,
                patch_valid=True,
                details=details,
            )

        # Step 3: Compute F2P and P2P scores
        f2p_score = self._compute_f2p(problem_data, test_result)
        p2p_score = self._compute_p2p(problem_data, test_result)

        details["f2p_score"] = f2p_score
        details["p2p_score"] = p2p_score

        # Step 4: Compute final reward
        resolved = f2p_score == 1.0 and p2p_score == 1.0

        if resolved:
            reward = 1.0
        else:
            # Partial credit: weighted combination
            reward = 0.6 * f2p_score + 0.4 * p2p_score

        reward *= self.reward_scale

        return RewardResult(
            reward=reward,
            resolved=resolved,
            f2p_score=f2p_score,
            p2p_score=p2p_score,
            patch_applied=True,
            patch_valid=True,
            details=details,
        )

    def _parse_completion(
        self,
        completion: str,
    ) -> Tuple[str, bool, str]:
        """
        Parse completion to extract patch.

        Returns:
            (patch_string, is_valid, error_message)
        """

        # Try SEARCH/REPLACE format first
        is_valid, edits = self.validator.validate_search_replace(completion)

        if is_valid and edits:
            # Convert to unified diff format
            patch = self._edits_to_patch(edits)
            valid, error = self.validator.validate_patch_format(patch)
            if valid:
                return patch, True, ""
            else:
                return "", False, error

        # Try direct diff format
        if "@@" in completion and ("+++" in completion or "---" in completion):
            # Extract diff portion
            lines = completion.split("\n")
            diff_lines = []
            in_diff = False

            for line in lines:
                if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                    in_diff = True
                if in_diff:
                    diff_lines.append(line)
                    if line.strip() == "" and len(diff_lines) > 3:
                        # End of hunk
                        pass

            if diff_lines:
                patch = "\n".join(diff_lines)
                valid, error = self.validator.validate_patch_format(patch)
                if valid:
                    return patch, True, ""

        return "", False, "Could not parse completion as patch"

    def _edits_to_patch(self, edits: List[Dict[str, str]]) -> str:
        """Convert SEARCH/REPLACE edits to unified diff format."""

        import difflib

        patches = []

        for edit in edits:
            file_name = edit["file"]
            search = edit["search"]
            replace = edit["replace"]

            # Create diff
            search_lines = search.splitlines(keepends=True)
            replace_lines = replace.splitlines(keepends=True)

            diff = difflib.unified_diff(
                search_lines,
                replace_lines,
                fromfile=f"a/{file_name}",
                tofile=f"b/{file_name}",
            )

            patches.append("".join(diff))

        return "\n".join(patches)

    def _run_tests_with_patch(
        self,
        problem_data: Dict[str, Any],
        patch: str,
    ) -> TestResult:
        """Apply patch and run tests."""

        repo_path = problem_data.get("repo_path")
        test_command = problem_data.get("test_command", "pytest")

        if not repo_path:
            # Create temporary directory with repo structure
            repo_path = self._setup_temp_repo(problem_data)

        try:
            # Apply patch
            apply_result = subprocess.run(
                f"git apply --check - <<< '{patch}'",
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            if apply_result.returncode != 0:
                return TestResult(
                    success=False,
                    exit_code=-2,  # Special code for patch failure
                    stdout="",
                    stderr=f"Patch failed to apply: {apply_result.stderr}",
                    duration=0,
                    tests_passed=0,
                    tests_failed=0,
                    tests_total=0,
                    f2p_score=0.0,
                    p2p_score=0.0,
                )

            # Actually apply the patch
            subprocess.run(
                f"git apply - <<< '{patch}'",
                shell=True,
                cwd=repo_path,
                capture_output=True,
            )

            # Run tests
            return self.test_executor.execute_tests(
                repo_path=repo_path,
                test_command=test_command,
            )

        except Exception as e:
            return TestResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=0,
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
                f2p_score=0.0,
                p2p_score=0.0,
            )

    def _setup_temp_repo(self, problem_data: Dict[str, Any]) -> str:
        """Setup temporary repository for testing."""

        temp_dir = tempfile.mkdtemp(prefix="swebench_")

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True)

        # Write files from problem data
        file_contents = problem_data.get("file_contents", {})
        for file_path, content in file_contents.items():
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)

        # Initial commit
        subprocess.run(["git", "add", "."], cwd=temp_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=temp_dir,
            capture_output=True,
        )

        return temp_dir

    def _compute_f2p(
        self,
        problem_data: Dict[str, Any],
        test_result: TestResult,
    ) -> float:
        """
        Compute Fail-to-Pass score.

        F2P = (number of previously failing tests that now pass) / (total F2P tests)
        """

        fail_to_pass = problem_data.get("FAIL_TO_PASS", [])

        if not fail_to_pass:
            return 1.0  # No F2P tests, consider it passed

        # Parse which tests passed from output
        passed_tests = self._extract_passed_tests(test_result.stdout)

        f2p_passed = sum(1 for test in fail_to_pass if test in passed_tests)

        return f2p_passed / len(fail_to_pass)

    def _compute_p2p(
        self,
        problem_data: Dict[str, Any],
        test_result: TestResult,
    ) -> float:
        """
        Compute Pass-to-Pass score.

        P2P = (number of tests that should pass and still pass) / (total P2P tests)
        """

        pass_to_pass = problem_data.get("PASS_TO_PASS", [])

        if not pass_to_pass:
            return 1.0  # No P2P tests, consider it passed

        # Parse which tests passed from output
        passed_tests = self._extract_passed_tests(test_result.stdout)

        p2p_passed = sum(1 for test in pass_to_pass if test in passed_tests)

        return p2p_passed / len(pass_to_pass)

    def _extract_passed_tests(self, output: str) -> set:
        """Extract set of passed test names from output."""

        passed = set()

        # Pytest format: "test_name PASSED"
        for match in re.finditer(r'(\S+)\s+PASSED', output):
            passed.add(match.group(1))

        # Also try: "PASSED test_name"
        for match in re.finditer(r'PASSED\s+(\S+)', output):
            passed.add(match.group(1))

        return passed


def create_reward_function(
    config: Dict[str, Any],
) -> callable:
    """
    Factory function to create reward function.

    Args:
        config: Configuration dict with reward settings

    Returns:
        Callable reward function
    """

    reward_config = config.get("reward", {})

    computer = RewardComputer(
        timeout=reward_config.get("timeout", 300),
        reward_scale=reward_config.get("scale", 1.0),
        penalty_invalid_patch=reward_config.get("penalty_invalid_patch", -0.5),
        penalty_failed_apply=reward_config.get("penalty_failed_apply", -0.3),
    )

    def reward_fn(problem_data: Dict[str, Any], completion: str) -> float:
        return computer.compute_reward(problem_data, completion)

    return reward_fn


if __name__ == "__main__":
    # Test reward computation
    logging.basicConfig(level=logging.INFO)

    computer = RewardComputer()

    # Test with mock data
    problem_data = {
        "instance_id": "test-001",
        "FAIL_TO_PASS": ["test_fix"],
        "PASS_TO_PASS": ["test_existing"],
    }

    completion = """
### src/module.py
<<<<<<< SEARCH
def broken_function():
    return 1 + 1
=======
def broken_function():
    return 2 + 2
>>>>>>> REPLACE
"""

    result = computer.evaluate_completion(problem_data, completion)
    print(f"Reward: {result.reward}")
    print(f"Resolved: {result.resolved}")
    print(f"Details: {result.details}")
