#!/bin/bash
#
# Build Docker image for secure code verification
#
# Usage:
#   ./build_docker.sh              # Build with default tag
#   ./build_docker.sh custom:tag   # Build with custom tag
#

set -e  # Exit on error

# Default configuration
DEFAULT_TAG="nexus-code-verifier:latest"
TAG="${1:-$DEFAULT_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Building Docker image for NEXUS code verification...${NC}"
echo "Tag: $TAG"
echo "Context: $SCRIPT_DIR"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed or not in PATH${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker daemon is not running${NC}"
    echo "Please start Docker daemon first"
    exit 1
fi

# Build the image
echo -e "${YELLOW}Building Docker image (this may take 2-5 minutes)...${NC}"
docker build \
    --tag "$TAG" \
    --file Dockerfile \
    --progress=plain \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
    echo ""
    echo "Image details:"
    docker images "$TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""
    echo -e "${GREEN}Verifying image...${NC}"

    # Test Python
    echo -n "  Python 3.11: "
    if docker run --rm "$TAG" python3 --version > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Test Node.js
    echo -n "  Node.js 20:  "
    if docker run --rm "$TAG" node --version > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Test SQLite
    echo -n "  SQLite:      "
    if docker run --rm "$TAG" sqlite3 --version > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Test Bash
    echo -n "  Bash:        "
    if docker run --rm "$TAG" bash --version > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    echo ""
    echo -e "${GREEN}Quick test (2+2=4):${NC}"
    docker run --rm "$TAG" python3 -c "print(2+2)"

    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "  1. Configure verifier: Edit scripts/bestofn/verifier_config.yaml"
    echo "  2. Run verifiers: python scripts/bestofn/generate_best_of_n.py"
    echo "  3. See documentation: scripts/bestofn/verifiers/README.md"
    echo ""
    echo "To remove the image later:"
    echo "  docker rmi $TAG"

else
    echo -e "${RED}✗ Docker build failed${NC}"
    echo "Check the error messages above for details"
    exit 1
fi
