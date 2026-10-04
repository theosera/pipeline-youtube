"""Keep accidental real downloads and external connections out of pytest.

Mocks still work: the audit hook runs only when Python reaches a real OS call.
It covers collection and fixture setup/teardown as well as test bodies/threads.
This is an accident guard, not an OS sandbox: arbitrary native code and network
activity inside other child programs are not intercepted.

Integration tests must explicitly use ``@pytest.mark.allow_real_network`` and/or
``@pytest.mark.allow_real_yt_dlp`` and be selected with
``pytest --run-real-integration -m allow_real_network`` (or allow_real_yt_dlp).
Marked tests are skipped otherwise. Never use the flag for the ordinary suite.
The yt-dlp permission allows a child process, including its network activity;
the network permission alone does not allow starting yt-dlp.
"""

import ipaddress
import os
import re
import socket
import sys
from collections.abc import Generator
from functools import wraps
from typing import Any

import pytest

# Match literal executables/modules, including shell/docker command arguments.
# Do not print command arguments or destinations: they can contain credentials.
_YT_DLP = re.compile(r"(?<![\w-])yt[-_]dlp(?:\.exe)?(?![\w-])", re.IGNORECASE)
_PROCESS_EVENTS = {"subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn", "os.system"}
_SPAWN_FUNCTIONS = (
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
)
_LOOKUP_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}
_CONNECTION_EVENTS = {"socket.connect", "socket.sendto", "socket.sendmsg"}


def _is_loopback(host: Any) -> bool:
    if isinstance(host, bytes):
        host = os.fsdecode(host)
    if not isinstance(host, str):
        return False
    if host.lower().rstrip(".") == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


class _TestIsolation:
    enabled = True
    allow_network = False
    allow_yt_dlp = False

    def check_command(self, command: Any) -> None:
        if self.enabled and not self.allow_yt_dlp and _YT_DLP.search(str(command)):
            pytest.fail("test isolation: blocked yt-dlp startup; mock the process", pytrace=False)

    def wrap_spawn(self, spawn: Any) -> Any:
        @wraps(spawn)
        def checked(mode: int, file: Any, *args: Any, **kwargs: Any) -> Any:
            # POSIX spawn* forks first and catches failures from exec in the
            # child, returning 127. Reject in the parent before that fork.
            if spawn.__name__.startswith("spawnv"):
                argv = args[0] if args else kwargs.get("args", ())
            else:
                argv = args[:-1] if spawn.__name__.endswith("e") else args
            self.check_command((file, argv))  # Never inspect the environment.
            return spawn(mode, file, *args, **kwargs)

        return checked

    def check_host(self, host: Any) -> None:
        if self.enabled and not self.allow_network and not _is_loopback(host):
            # pytest's failure bypasses application except Exception/OSError.
            pytest.fail(
                "test isolation: blocked external network; mock the connection", pytrace=False
            )

    def check_address(self, connection: socket.socket, address: Any) -> None:
        if connection.family in (socket.AF_INET, socket.AF_INET6) and address is not None:
            self.check_host(address[0])

    def wrap_socket_method(self, method: Any) -> Any:
        @wraps(method)
        def checked(connection: socket.socket, *args: Any, **kwargs: Any) -> Any:
            # CPython can resolve hostnames while parsing a sockaddr, BEFORE
            # emitting socket.connect/sendto/sendmsg. Check at the Python entry
            # too, so direct connect((hostname, port)) cannot leak a DNS query.
            if method.__name__ == "sendmsg":
                address = args[3] if len(args) > 3 else kwargs.get("address")
            else:
                address = args[-1]
            self.check_address(connection, address)
            return method(connection, *args, **kwargs)

        return checked

    def wrap_lookup(self, lookup: Any) -> Any:
        @wraps(lookup)
        def checked(host: Any, *args: Any, **kwargs: Any) -> Any:
            address = host[0] if lookup.__name__ == "getnameinfo" else host
            if address is not None:
                self.check_host(address)
            return lookup(host, *args, **kwargs)

        return checked

    def audit(self, event: str, args: tuple[Any, ...]) -> None:
        if not self.enabled:
            return
        if event in _PROCESS_EVENTS and not self.allow_yt_dlp:
            # Ignore cwd/env; only inspect the executable and command arguments.
            command = args[:1] if event == "os.system" else args[:2]
            if event == "os.spawn":
                command = args[1:3]
            self.check_command(command)
        if self.allow_network:
            return
        host = None
        check_host = False
        if event in _LOOKUP_EVENTS:
            host = args[0]
            # getaddrinfo(None, ...) constructs local bind addresses without DNS.
            check_host = host is not None
        elif event == "socket.getnameinfo":
            host = args[0][0]
            check_host = True
        elif event in _CONNECTION_EVENTS:
            connection, address = args
            if connection.family in (socket.AF_INET, socket.AF_INET6) and address is not None:
                host = address[0]
                check_host = True
        if check_host and not _is_loopback(host):
            self.check_host(host)


_isolation: _TestIsolation | None = None
_patches: pytest.MonkeyPatch | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-real-integration",
        action="store_true",
        default=False,
        help="Explicitly enable tests marked allow_real_network/allow_real_yt_dlp",
    )


def pytest_configure(config: pytest.Config) -> None:
    global _isolation, _patches
    _isolation = _TestIsolation()
    sys.addaudithook(_isolation.audit)
    _patches = pytest.MonkeyPatch()
    for name in _SPAWN_FUNCTIONS:
        if hasattr(os, name):
            _patches.setattr(os, name, _isolation.wrap_spawn(getattr(os, name)))
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):
            method = getattr(socket.socket, name)
            _patches.setattr(socket.socket, name, _isolation.wrap_socket_method(method))
    for name in (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        _patches.setattr(socket, name, _isolation.wrap_lookup(getattr(socket, name)))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-real-integration"):
        skip = pytest.mark.skip(reason="real integration requires --run-real-integration")
        for item in items:
            if item.get_closest_marker("allow_real_network") or item.get_closest_marker(
                "allow_real_yt_dlp"
            ):
                item.add_marker(skip)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator:
    assert _isolation is not None
    enabled = item.config.getoption("--run-real-integration")
    _isolation.allow_network = enabled and bool(item.get_closest_marker("allow_real_network"))
    _isolation.allow_yt_dlp = enabled and bool(item.get_closest_marker("allow_real_yt_dlp"))
    try:
        return (yield)
    finally:
        _isolation.allow_network = _isolation.allow_yt_dlp = False


def pytest_unconfigure(config: pytest.Config) -> None:
    # Audit hooks cannot be removed; leave this instance inert after pytest.
    if _isolation is not None:
        _isolation.enabled = False
    if _patches is not None:
        _patches.undo()
