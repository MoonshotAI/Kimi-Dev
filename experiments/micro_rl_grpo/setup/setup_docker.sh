#!/bin/bash
# ============================================================================
# Setup Docker Sandbox Environment for Test Execution
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  Setting up Docker Sandbox Environment"
echo "=============================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Docker installation
check_docker() {
    log_info "Checking Docker installation..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running. Please start Docker."
        exit 1
    fi

    docker --version
    log_info "Docker is available"
}

# Create base Dockerfile for SWE-bench execution
create_dockerfile() {
    log_info "Creating Dockerfile for SWE-bench sandbox..."

    mkdir -p "$PROJECT_DIR/docker"

    cat > "$PROJECT_DIR/docker/Dockerfile.swebench" << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install common Python testing tools
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov \
    pytest-timeout \
    pytest-xdist \
    coverage \
    hypothesis \
    mock \
    tox \
    nox

# Create workspace directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["/bin/bash"]
EOF

    log_info "Dockerfile created at $PROJECT_DIR/docker/Dockerfile.swebench"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."

    docker build \
        -t swebench-sandbox:latest \
        -f "$PROJECT_DIR/docker/Dockerfile.swebench" \
        "$PROJECT_DIR/docker"

    log_info "Docker image built successfully"
}

# Create docker-compose for multiple containers
create_docker_compose() {
    log_info "Creating docker-compose configuration..."

    cat > "$PROJECT_DIR/docker/docker-compose.yml" << 'EOF'
version: '3.8'

services:
  sandbox:
    image: swebench-sandbox:latest
    volumes:
      - sandbox_workspaces:/workspace
    networks:
      - sandbox_network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: false

networks:
  sandbox_network:
    driver: bridge

volumes:
  sandbox_workspaces:
EOF

    log_info "docker-compose.yml created"
}

# Create sandbox manager script
create_sandbox_manager() {
    log_info "Creating sandbox manager..."

    cat > "$PROJECT_DIR/docker/sandbox_manager.py" << 'EOF'
"""
Docker Sandbox Manager for SWE-bench Test Execution
"""

import docker
import tempfile
import os
import shutil
import logging
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages Docker containers for safe test execution."""

    def __init__(
        self,
        image_name: str = "swebench-sandbox:latest",
        max_containers: int = 50,
        timeout: int = 300,
        memory_limit: str = "4g",
        cpu_limit: float = 2.0
    ):
        self.client = docker.from_env()
        self.image_name = image_name
        self.max_containers = max_containers
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.active_containers: Dict[str, docker.models.containers.Container] = {}

    def _ensure_image(self):
        """Ensure Docker image exists."""
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            logger.error(f"Image {self.image_name} not found. Run setup_docker.sh first.")
            raise

    def create_container(
        self,
        repo_path: str,
        container_id: Optional[str] = None
    ) -> Tuple[str, docker.models.containers.Container]:
        """Create a new sandbox container."""

        self._ensure_image()

        container_id = container_id or str(uuid.uuid4())[:8]

        container = self.client.containers.create(
            self.image_name,
            name=f"swebench-{container_id}",
            volumes={
                repo_path: {"bind": "/workspace/repo", "mode": "rw"}
            },
            mem_limit=self.memory_limit,
            cpu_quota=int(self.cpu_limit * 100000),
            network_mode="none",  # Isolated network
            security_opt=["no-new-privileges:true"],
            detach=True,
            tty=True,
        )

        self.active_containers[container_id] = container
        return container_id, container

    def execute_tests(
        self,
        container_id: str,
        test_command: str,
        cwd: str = "/workspace/repo"
    ) -> Tuple[int, str, str]:
        """Execute tests in container and return results."""

        container = self.active_containers.get(container_id)
        if not container:
            raise ValueError(f"Container {container_id} not found")

        try:
            container.start()

            # Execute test command
            exec_result = container.exec_run(
                cmd=f"/bin/bash -c 'cd {cwd} && {test_command}'",
                workdir=cwd,
                demux=True,
            )

            exit_code = exec_result.exit_code
            stdout = exec_result.output[0].decode() if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode() if exec_result.output[1] else ""

            return exit_code, stdout, stderr

        except Exception as e:
            logger.error(f"Error executing tests: {e}")
            return -1, "", str(e)

    def apply_patch(
        self,
        container_id: str,
        patch_content: str,
        cwd: str = "/workspace/repo"
    ) -> bool:
        """Apply a patch to the repository in container."""

        container = self.active_containers.get(container_id)
        if not container:
            raise ValueError(f"Container {container_id} not found")

        try:
            container.start()

            # Write patch to file
            patch_cmd = f"cat > /tmp/patch.diff << 'PATCH_EOF'\n{patch_content}\nPATCH_EOF"
            container.exec_run(cmd=f"/bin/bash -c \"{patch_cmd}\"")

            # Apply patch
            apply_result = container.exec_run(
                cmd=f"/bin/bash -c 'cd {cwd} && git apply /tmp/patch.diff'",
                workdir=cwd,
            )

            return apply_result.exit_code == 0

        except Exception as e:
            logger.error(f"Error applying patch: {e}")
            return False

    def cleanup_container(self, container_id: str):
        """Remove a container."""
        container = self.active_containers.pop(container_id, None)
        if container:
            try:
                container.stop(timeout=5)
                container.remove(force=True)
            except Exception as e:
                logger.warning(f"Error cleaning up container {container_id}: {e}")

    def cleanup_all(self):
        """Remove all active containers."""
        for container_id in list(self.active_containers.keys()):
            self.cleanup_container(container_id)

    def run_evaluation(
        self,
        repo_path: str,
        patch_content: str,
        test_command: str,
        setup_command: Optional[str] = None
    ) -> Dict:
        """Full evaluation: apply patch and run tests."""

        result = {
            "success": False,
            "patch_applied": False,
            "tests_passed": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "duration": 0,
        }

        start_time = time.time()
        container_id = None

        try:
            # Create container
            container_id, _ = self.create_container(repo_path)

            # Run setup if provided
            if setup_command:
                exit_code, _, stderr = self.execute_tests(
                    container_id, setup_command
                )
                if exit_code != 0:
                    result["stderr"] = f"Setup failed: {stderr}"
                    return result

            # Apply patch
            patch_applied = self.apply_patch(container_id, patch_content)
            result["patch_applied"] = patch_applied

            if not patch_applied:
                result["stderr"] = "Failed to apply patch"
                return result

            # Run tests
            exit_code, stdout, stderr = self.execute_tests(
                container_id, test_command
            )

            result["exit_code"] = exit_code
            result["stdout"] = stdout
            result["stderr"] = stderr
            result["tests_passed"] = exit_code == 0
            result["success"] = exit_code == 0

        except Exception as e:
            result["stderr"] = str(e)

        finally:
            if container_id:
                self.cleanup_container(container_id)
            result["duration"] = time.time() - start_time

        return result


def test_sandbox():
    """Test sandbox functionality."""
    manager = SandboxManager(max_containers=5)

    # Create a test repo
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize git repo
        os.system(f"cd {tmpdir} && git init")

        # Create test file
        with open(os.path.join(tmpdir, "test_example.py"), "w") as f:
            f.write("""
def test_pass():
    assert 1 + 1 == 2

def test_fail():
    assert 1 + 1 == 3
""")

        os.system(f"cd {tmpdir} && git add . && git commit -m 'initial'")

        # Test execution
        result = manager.run_evaluation(
            repo_path=tmpdir,
            patch_content="",
            test_command="pytest test_example.py -v",
        )

        print(f"Result: {result}")

    manager.cleanup_all()


if __name__ == "__main__":
    test_sandbox()
EOF

    log_info "Sandbox manager created"
}

# Test Docker setup
test_docker() {
    log_info "Testing Docker setup..."

    # Run a simple test
    docker run --rm swebench-sandbox:latest python -c "print('Docker sandbox working!')"

    log_info "Docker test passed"
}

# Main
main() {
    check_docker
    create_dockerfile
    build_image
    create_docker_compose
    create_sandbox_manager
    test_docker

    echo ""
    echo "=============================================="
    echo "  Docker sandbox setup complete!"
    echo "=============================================="
    echo ""
    echo "To test the sandbox:"
    echo "  python docker/sandbox_manager.py"
    echo ""
}

main "$@"
