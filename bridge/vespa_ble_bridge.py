"""Explicit custom BLE v2 protocol. Requires firmware/hockey_v2, NOT stock firmware."""

from __future__ import annotations
import asyncio
import binascii
import struct
import time
from core.actuators import action_to_pwm

UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
PROTOCOL_ID = b"HOCKEY-V2"


def encode_command(sequence, left, right, ttl_ms=150):
    if not 0 <= sequence <= 65535 or not 20 <= ttl_ms <= 200:
        raise ValueError("Invalid sequence/TTL")
    if not -255 <= left <= 255 or not -255 <= right <= 255:
        raise ValueError("PWM out of range")
    payload = struct.pack("<2sBHhhH", b"\xaa\x55", 2, sequence, left, right, ttl_ms)
    return payload + struct.pack("<H", binascii.crc_hqx(payload, 0xFFFF))


def decode_command(packet):
    if len(packet) != 13:
        raise ValueError("Invalid packet size")
    payload = packet[:-2]
    if binascii.crc_hqx(payload, 0xFFFF) != struct.unpack("<H", packet[-2:])[0]:
        raise ValueError("CRC mismatch")
    header, version, seq, left, right, ttl = struct.unpack("<2sBHhhH", payload)
    if header != b"\xaa\x55" or version != 2:
        raise ValueError("Wrong protocol")
    encode_command(seq, left, right, ttl)
    return seq, left, right, ttl


def apply_deadband_compensation(norm_val, deadband_pwm=31, max_pwm=255):
    return int(action_to_pwm([norm_val], deadband_pwm / max_pwm, trim=(1.0,))[0])


class VespaBluetoothBridge:
    UART_RX_CHAR_UUID = UART_RX_CHAR_UUID

    def __init__(self, target_mac=None, trim_left=1.0, trim_right=1.0, deadband=0.12, control_rate_hz=30.0):
        self.target_mac = target_mac
        self.trim = (trim_left, trim_right)
        self.deadband = deadband
        if not target_mac:
            raise ValueError("Set a unique BLE address; automatic first-device selection is disabled")
        self.control_rate_hz = control_rate_hz
        if not 0 < control_rate_hz <= 60 or not 0 <= deadband < 1:
            raise ValueError("Invalid radio rate or deadband")
        self.client = None
        self.is_connected = False
        self.sequence = 0
        self.last_send_time = -float("inf")
        self.last_write_ms = 0.0
        self.lock = asyncio.Lock()

    async def connect(self):
        from bleak import BleakClient

        self.client = BleakClient(self.target_mac)
        try:
            await asyncio.wait_for(self.client.connect(), 8.0)
            hello = bytes(await asyncio.wait_for(self.client.read_gatt_char(UART_TX_CHAR_UUID), 1.0))
            if hello != PROTOCOL_ID:
                raise ValueError("Firmware does not implement HOCKEY-V2")
            self.is_connected = True
            await self.emergency_stop()
            return True
        except Exception:
            if self.client.is_connected:
                await self.client.disconnect()
            self.is_connected = False
            raise

    async def send_wheel_speeds(self, left, right, *, deadline=None):
        if not self.is_connected or self.client is None:
            return False
        async with self.lock:
            wait = 1 / self.control_rate_hz - (time.monotonic() - self.last_send_time)
            if wait > 0:
                await asyncio.sleep(wait)
            now = time.monotonic()
            expired = deadline is not None and deadline - now <= 0.06
            if expired:
                left = right = 0.0
            # Reserve 50 ms for radio transport; GATT completion is not a motor RTT.
            ttl = max(20, min(150, int((deadline - now - 0.05) * 1000))) if deadline is not None else 150
            pwm = action_to_pwm([left, right], self.deadband, self.trim)
            packet = encode_command(self.sequence, int(pwm[0]), int(pwm[1]), ttl)
            self.sequence = (self.sequence + 1) & 65535
            try:
                start = time.monotonic()
                await asyncio.wait_for(
                    self.client.write_gatt_char(UART_RX_CHAR_UUID, packet, response=True), 0.10
                )
                self.last_write_ms = (time.monotonic() - start) * 1000
                self.last_send_time = time.monotonic()
                return not expired
            except (OSError, asyncio.TimeoutError):
                return False

    async def emergency_stop(self):
        if self.client is None or not self.is_connected:
            return
        packet = encode_command(self.sequence, 0, 0, 20)
        self.sequence = (self.sequence + 1) & 65535
        try:
            await asyncio.wait_for(
                self.client.write_gatt_char(UART_RX_CHAR_UUID, packet, response=True), 0.10
            )
        except Exception:
            pass  # Firmware TTL remains the independent fail-stop.

    async def disconnect(self):
        await self.emergency_stop()
        if self.client is not None and self.client.is_connected:
            await self.client.disconnect()
        self.is_connected = False


class VespaTeamBridge:
    def __init__(self, macs, trims=None, control_rate_hz=30.0):
        if len(macs) != 3 or len(set(macs)) != 3:
            raise ValueError("Configure three unique robot addresses")
        trims = trims or [(1.0, 1.0)] * 3
        self.robots = [
            VespaBluetoothBridge(m, trim_left=t[0], trim_right=t[1], control_rate_hz=control_rate_hz)
            for m, t in zip(macs, trims)
        ]

    async def connect_all(self):
        results = await asyncio.gather(*(r.connect() for r in self.robots), return_exceptions=True)
        if not all(result is True for result in results):
            await self.disconnect_all()
            raise RuntimeError("Not all robots completed the HOCKEY-V2 handshake")
        return results

    async def send_team_actions(self, actions, deadline=None):
        if len(actions) != 3:
            raise ValueError("Expected three robot actions")
        results = await asyncio.gather(
            *(r.send_wheel_speeds(*a, deadline=deadline) for r, a in zip(self.robots, actions)),
            return_exceptions=True,
        )
        if not all(v is True for v in results):
            await self.emergency_stop_all()
            return False
        return True

    async def emergency_stop_all(self):
        await asyncio.gather(*(r.emergency_stop() for r in self.robots), return_exceptions=True)

    async def disconnect_all(self):
        await asyncio.gather(*(r.disconnect() for r in self.robots), return_exceptions=True)
