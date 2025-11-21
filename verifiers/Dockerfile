# Nexus Code Verifier Docker Image
# Multi-language secure execution environment for Python, JavaScript, Bash, SQL
#
# Build: docker build -t nexus-code-verifier:latest -f scripts/bestofn/verifiers/Dockerfile .
# Size target: <500MB

FROM python:3.11-slim

LABEL maintainer="nexus-team"
LABEL description="Secure multi-language code execution sandbox"
LABEL version="0.2.0"

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================
# Install System Dependencies
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Node.js 20.x
    curl \
    gnupg \
    ca-certificates \
    # SQLite
    sqlite3 \
    # Bash (already included but ensure it's there)
    bash \
    # Utilities
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Install Python Packages (Common for Testing)
# ============================================================================

RUN pip install --no-cache-dir \
    # Testing frameworks
    pytest==7.4.3 \
    # Data science (common in code challenges)
    numpy==1.26.2 \
    # No other packages to keep size small
    && pip cache purge

# ============================================================================
# Install Node.js Packages (Common for Testing)
# ============================================================================

RUN npm install -g \
    # Testing framework
    jest@29.7.0 \
    # Keep global packages minimal
    && npm cache clean --force

# ============================================================================
# Security Configuration
# ============================================================================

# Create non-root user
RUN groupadd -r sandbox -g 1000 && \
    useradd -r -g sandbox -u 1000 -m -s /bin/bash sandbox

# Create writable temp directory (will be mounted with tmpfs)
RUN mkdir -p /tmp && chmod 1777 /tmp

# Remove unnecessary setuid binaries to prevent privilege escalation
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# ============================================================================
# Restricted Bash Configuration
# ============================================================================

# Create restricted bash profile that limits dangerous commands
RUN echo '# Restricted bash environment' > /home/sandbox/.bashrc && \
    echo 'export PATH=/usr/local/bin:/usr/bin:/bin' >> /home/sandbox/.bashrc && \
    echo 'ulimit -t 10   # 10 second CPU limit' >> /home/sandbox/.bashrc && \
    echo 'ulimit -m 524288  # 512MB memory limit' >> /home/sandbox/.bashrc && \
    echo 'ulimit -f 10240   # 10MB file size limit' >> /home/sandbox/.bashrc && \
    chown sandbox:sandbox /home/sandbox/.bashrc

# ============================================================================
# Environment Configuration
# ============================================================================

# Set working directory
WORKDIR /tmp

# Switch to non-root user
USER sandbox

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NODE_ENV=production \
    HOME=/home/sandbox

# ============================================================================
# Health Check
# ============================================================================

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "print('ok')" && node -e "console.log('ok')" && sqlite3 --version

# ============================================================================
# Default Command
# ============================================================================

# Keep container alive (will be overridden by exec commands)
CMD ["tail", "-f", "/dev/null"]

# ============================================================================
# Image Metadata
# ============================================================================

# Document exposed capabilities
LABEL capabilities="python3,node,sqlite3,bash"
LABEL security.network="disabled"
LABEL security.filesystem="read-only-root"
LABEL security.user="non-root"

# Size optimization notes:
# - Using python:3.11-slim (not alpine due to compatibility issues)
# - Minimal package installation
# - No build tools retained
# - Cache purging after installations
# Expected size: ~400-500MB
