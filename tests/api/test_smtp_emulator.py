"""
Tests for the SMTP emulator TCP handler.

Each test instantiates ``SmtpEmulator`` with mocked ``GenPotEngine`` and
``StateManager``, then drives ``handle_client`` with fake ``StreamReader``
/ ``StreamWriter`` objects so we never open real sockets.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from server.core.models import UnifiedResponse
from server.emulators.smtp_emulator import SmtpEmulator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_reader(lines: list[bytes]) -> AsyncMock:
    """Return a mock StreamReader that yields *lines* then EOF."""
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.readline = AsyncMock(side_effect=lines + [b""])
    return reader


def _make_writer() -> MagicMock:
    """Return a mock StreamWriter that records writes."""
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.get_extra_info = MagicMock(return_value=("127.0.0.1", 54321))
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    return writer


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.process = AsyncMock()
    return engine


@pytest.fixture
def mock_state_manager():
    sm = MagicMock()
    sm.clear_session = MagicMock()
    return sm


@pytest.fixture
def emulator(mock_engine, mock_state_manager):
    return SmtpEmulator(mock_engine, mock_state_manager)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_greeting_sent_on_connect(emulator, mock_engine):
    """The 220 banner must be the very first thing written to the socket."""
    mock_engine.process.return_value = UnifiedResponse(status_code=200, data="250 OK")
    reader = _make_reader([b"QUIT\r\n"])
    writer = _make_writer()

    await emulator.handle_client(reader, writer)

    first_write = writer.write.call_args_list[0]
    assert first_write == call(b"220 ESMTP GenPot Honeypot\r\n")


@pytest.mark.asyncio
async def test_standard_command_forwarded(emulator, mock_engine):
    """A plain EHLO command should produce a correctly mapped UnifiedRequest."""
    mock_engine.process.return_value = UnifiedResponse(
        status_code=200, data="250 Hello"
    )
    reader = _make_reader([b"EHLO example.com\r\n"])
    writer = _make_writer()

    await emulator.handle_client(reader, writer)

    req = mock_engine.process.call_args[0][0]
    assert req.protocol == "smtp"
    assert req.command == "EHLO example.com"
    assert req.method == "SMTP"
    assert req.path == "/smtp"
    assert req.source_ip == "127.0.0.1"
    assert req.session_id is not None

    # Verify the engine's reply was written back.
    assert call(b"250 Hello\r\n") in writer.write.call_args_list


@pytest.mark.asyncio
async def test_data_mode_transition(emulator, mock_engine):
    """
    When the engine answers DATA with a 354, the emulator must enter data
    mode, buffer subsequent lines, and forward a [DATA PAYLOAD] request
    when the lone '.' terminator arrives.
    """
    mock_engine.process.side_effect = [
        UnifiedResponse(status_code=200, data="354 Start mail input"),
        UnifiedResponse(status_code=200, data="250 Message accepted"),
    ]
    reader = _make_reader(
        [
            b"DATA\r\n",
            b"Subject: Test\r\n",
            b"Hello world\r\n",
            b".\r\n",
        ]
    )
    writer = _make_writer()

    await emulator.handle_client(reader, writer)

    # Two calls: DATA command, then the buffered payload.
    assert mock_engine.process.call_count == 2

    payload_req = mock_engine.process.call_args_list[1][0][0]
    assert payload_req.command == "[DATA PAYLOAD]"
    assert "Subject: Test" in payload_req.body
    assert "Hello world" in payload_req.body


@pytest.mark.asyncio
async def test_quit_closes_connection(emulator, mock_engine):
    """After QUIT the emulator must send the response and stop the loop."""
    mock_engine.process.side_effect = [
        UnifiedResponse(status_code=200, data="250 Hello"),
        UnifiedResponse(status_code=200, data="221 Bye"),
    ]
    reader = _make_reader([b"EHLO test\r\n", b"QUIT\r\n"])
    writer = _make_writer()

    await emulator.handle_client(reader, writer)

    # "221 Bye" should have been written.
    assert call(b"221 Bye\r\n") in writer.write.call_args_list
    # Writer must be closed.
    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_session_cleanup_on_disconnect(emulator, mock_state_manager, mock_engine):
    """clear_session must always be called, even on unexpected disconnect."""
    mock_engine.process.return_value = UnifiedResponse(status_code=200, data="250 OK")
    reader = _make_reader([b"EHLO x\r\n"])  # EOF after one command
    writer = _make_writer()

    await emulator.handle_client(reader, writer)

    mock_state_manager.clear_session.assert_called_once()


@pytest.mark.asyncio
async def test_eof_handling(emulator, mock_state_manager, mock_engine):
    """An immediate EOF (empty read) must break cleanly without errors."""
    reader = _make_reader([])  # immediate EOF
    writer = _make_writer()

    await emulator.handle_client(reader, writer)

    # Engine should never have been called.
    mock_engine.process.assert_not_called()
    # Session cleanup still happens.
    mock_state_manager.clear_session.assert_called_once()
    writer.close.assert_called_once()
