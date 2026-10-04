"""Inert regression controls for the default test isolation policy.

Each probe runs in a child pytest. Raw socket sentinels are installed before
conftest configures its guard; a second audit hook is registered afterwards.
Thus even a missing guard cannot reach native DNS/network or start a tool.
"""

import os
import socket

import pytest

pytest_plugins = ["pytester"]

_PRE_GUARD = """
import functools
import socket
import sys
import pytest

class IndependentInterception(BaseException):
    pass

def independent_socket_stop(original):
    @functools.wraps(original)
    def stopped(*args, **kwargs):
        raise IndependentInterception("independent socket backstop: " + original.__name__)
    return stopped

for name in ("connect", "connect_ex", "sendto", "sendmsg"):
    if hasattr(socket.socket, name):
        setattr(socket.socket, name, independent_socket_stop(getattr(socket.socket, name)))
for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
    setattr(socket, name, independent_socket_stop(getattr(socket, name)))
"""

_BASE_COMPATIBILITY = """
# Only for before-guard main, which has no conftest. Accept the new option
# and marker names without supplying any skip, permission or guard policy.
def pytest_addoption(parser):
    parser.addoption("--run-real-integration", action="store_true", default=False)

def pytest_configure(config):
    config.addinivalue_line("markers", "allow_real_network: base verification adapter")
    config.addinivalue_line("markers", "allow_real_yt_dlp: base verification adapter")
"""

_POST_GUARD = """
def pytest_sessionstart(session):
    def independent_audit_stop(event, args):
        if event in {
            "subprocess.Popen", "os.fork", "os.exec", "os.posix_spawn", "os.spawn", "os.system",
            "socket.connect", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
            "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo",
        }:
            raise IndependentInterception("independent audit backstop: " + event)
    sys.addaudithook(independent_audit_stop)
"""

_PROBE_IMPORTS = """
import os
import socket
import subprocess
import sys
import pytest
from conftest import IndependentInterception
"""


@pytest.fixture
def probe_project(pytester: pytest.Pytester, pytestconfig: pytest.Config):
    root = pytestconfig.rootpath
    candidate_path = root / "tests" / "conftest.py"
    # Missing on before-guard main. Do not mask a deleted hook in an existing
    # candidate: the parser-only adapter is used solely when the file is absent.
    candidate = candidate_path.read_text() if candidate_path.exists() else _BASE_COMPATIBILITY
    config = (root / "pyproject.toml").read_text()
    pytester.makepyprojecttoml(config)

    def prepare(source: str, *, hooks: str = "") -> pytest.Pytester:
        pytester.makeconftest(_PRE_GUARD + candidate + _POST_GUARD + hooks)
        pytester.makepyfile(test_probe=_PROBE_IMPORTS + source)
        return pytester

    return prepare


def test_process_and_socket_boundaries(probe_project):
    project = probe_project("""
@pytest.mark.parametrize("command", [
    ["yt-dlp", "--version"],
    [sys.executable, "-m", "yt_dlp", "--version"],
    ["YT-DLP.EXE", "--version"],
    ["Yt-Dlp", "--version"],
    ["YT_DLP", "--version"],
])
def test_launcher(command):
    with pytest.raises(pytest.fail.Exception, match="test isolation: blocked yt-dlp"):
        subprocess.Popen(command)

@pytest.mark.parametrize("name", [
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
])
def test_spawn_fails_in_parent_before_fork(name):
    if not hasattr(os, name):
        pytest.skip("spawn variant unavailable")
    argv = ["yt-dlp", "--version"]
    args = [os.P_WAIT, "yt-dlp"]
    args.extend([argv] if name.startswith("spawnv") else argv)
    if name.endswith("e"):
        args.append({})
    with pytest.raises(pytest.fail.Exception, match="test isolation: blocked yt-dlp"):
        getattr(os, name)(*args)

@pytest.mark.parametrize("method", ["connect", "connect_ex", "sendto", "sendmsg"])
def test_hostname_before_native_resolution(method):
    if not hasattr(socket.socket, method):
        pytest.skip("socket method unavailable")
    kind = socket.SOCK_DGRAM if method.startswith("send") else socket.SOCK_STREAM
    with socket.socket(type=kind) as connection:
        with pytest.raises(pytest.fail.Exception, match="test isolation: blocked external network"):
            if method == "sendto":
                connection.sendto(b"inert", ("example.invalid", 443))
            elif method == "sendmsg":
                connection.sendmsg([b"inert"], [], 0, ("example.invalid", 443))
            else:
                getattr(connection, method)(("example.invalid", 443))

def test_numeric_connection():
    with socket.socket() as connection:
        with pytest.raises(pytest.fail.Exception, match="test isolation: blocked external network"):
            connection.connect(("192.0.2.1", 443))

def test_external_dns():
    with pytest.raises(pytest.fail.Exception, match="test isolation: blocked external network"):
        socket.getaddrinfo("example.invalid", 443)

def test_loopback_reaches_independent_backstop():
    with socket.socket() as connection:
        with pytest.raises(IndependentInterception, match="independent socket backstop: connect"):
            connection.connect(("127.0.0.1", 443))
""")
    result = project.runpytest_subprocess("-q", "--tb=short", timeout=30)
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    unavailable = sum(
        not hasattr(os, name)
        for name in (
            "spawnl",
            "spawnle",
            "spawnlp",
            "spawnlpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
        )
    ) + int(not hasattr(socket.socket, "sendmsg"))
    result.assert_outcomes(passed=20 - unavailable, skipped=unavailable)


def test_flag_alone_keeps_unmarked_tests_guarded(probe_project):
    project = probe_project("""
def test_unmarked():
    with pytest.raises(pytest.fail.Exception, match="blocked yt-dlp"):
        sys.audit("subprocess.Popen", "yt-dlp", ["yt-dlp"], None, None)
    with pytest.raises(pytest.fail.Exception, match="blocked external network"):
        sys.audit("socket.getaddrinfo", "example.invalid", 443, 0, 0, 0)
""")
    result = project.runpytest_subprocess("-q", "--run-real-integration", timeout=30)
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    result.assert_outcomes(passed=1)


def test_real_markers_require_explicit_flag(probe_project):
    project = probe_project("""
@pytest.mark.allow_real_network
def test_network():
    pytest.fail("marked network test ran without explicit flag")

@pytest.mark.allow_real_yt_dlp
def test_tool():
    pytest.fail("marked tool test ran without explicit flag")
""")
    result = project.runpytest_subprocess("-q", timeout=30)
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    result.assert_outcomes(skipped=2)


def test_marker_permissions_fixtures_and_reset(probe_project):
    project = probe_project(
        """
def check_permissions(network, tool):
    expected = IndependentInterception if network else pytest.fail.Exception
    with pytest.raises(expected):
        sys.audit("socket.getaddrinfo", "example.invalid", 443, 0, 0, 0)
    expected = IndependentInterception if tool else pytest.fail.Exception
    with pytest.raises(expected):
        sys.audit("subprocess.Popen", "yt-dlp", ["yt-dlp"], None, None)

@pytest.fixture
def permission_scope(request):
    permissions = (
        bool(request.node.get_closest_marker("allow_real_network")),
        bool(request.node.get_closest_marker("allow_real_yt_dlp")),
    )
    check_permissions(*permissions)
    yield
    check_permissions(*permissions)

@pytest.mark.allow_real_network
def test_01_network(permission_scope):
    check_permissions(True, False)

def test_02_unmarked_after_network(permission_scope):
    check_permissions(False, False)

@pytest.mark.allow_real_yt_dlp
def test_03_tool(permission_scope):
    check_permissions(False, True)

def test_04_unmarked_after_tool(permission_scope):
    check_permissions(False, False)

@pytest.mark.allow_real_network
@pytest.mark.allow_real_yt_dlp
def test_05_both(permission_scope):
    check_permissions(True, True)
""",
        hooks="""
def pytest_sessionfinish(session, exitstatus):
    # The final test has both permissions: test protocol teardown must reset
    # them even before a next item gets a chance to assign new permissions.
    with pytest.raises(pytest.fail.Exception, match="blocked external network"):
        sys.audit("socket.getaddrinfo", "example.invalid", 443, 0, 0, 0)
    with pytest.raises(pytest.fail.Exception, match="blocked yt-dlp"):
        sys.audit("subprocess.Popen", "yt-dlp", ["yt-dlp"], None, None)
""",
    )
    result = project.runpytest_subprocess("-q", "--run-real-integration", "--tb=short", timeout=30)
    assert result.ret == pytest.ExitCode.OK, result.stdout.str()
    result.assert_outcomes(passed=5)


def test_worker_violation_fails_pytest(probe_project):
    project = probe_project("""
import threading

def test_worker():
    observed = []
    def attempt():
        try:
            subprocess.Popen(["yt-dlp", "--version"])
        except BaseException as exc:
            observed.append(type(exc).__name__ + ": " + str(exc))
            raise
    worker = threading.Thread(target=attempt)
    worker.start()
    worker.join()
    print(observed)
    assert observed == ["Failed: test isolation: blocked yt-dlp startup; mock the process"]
""")
    result = project.runpytest_subprocess("-q", "--tb=short", timeout=30)
    assert result.ret == pytest.ExitCode.TESTS_FAILED, result.stdout.str()
    result.assert_outcomes(failed=1)
    output = result.stdout.str()
    assert "PytestUnhandledThreadExceptionWarning" in output
    assert "test isolation: blocked yt-dlp startup" in output
    assert "IndependentInterception" not in output
