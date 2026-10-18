"""
Ponte de Comunicação Bluetooth BLE com o RoboCore HockeyBot (Placa Vespa / ESP32).

Permite que a rede neural treinada no simulador envie comandos de velocidade
diretamente para os carrinhos físicos reais via Bluetooth BLE a partir do Python.
Compatível com a biblioteca `bleak`.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Optional

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False


class VespaBluetoothBridge:
    """
    Gerenciador de conexão BLE com a placa RoboCore Vespa (ESP32).
    Nome padrão do dispositivo: 'HockeyBot-XX:XX' ou 'Vespa-XX:XX'.
    """

    # UUID padrão da característica GATT usada para controle
    UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

    def __init__(self, device_name_prefix: str = "HockeyBot"):
        self.device_name_prefix = device_name_prefix
        self.client: Optional[BleakClient] = None
        self.is_connected: bool = False

    async def scan_and_connect(self, timeout: float = 6.0) -> bool:
        if not BLEAK_AVAILABLE:
            print("[Vespa BLE] Biblioteca 'bleak' não instalada. Execute: pip install bleak")
            return False

        print(f"[Vespa BLE] Procurando dispositivos com prefixo '{self.device_name_prefix}'...")
        devices = await BleakScanner.discover(timeout=timeout)
        target = None
        for d in devices:
            if d.name and (self.device_name_prefix in d.name or "Vespa" in d.name):
                target = d
                break

        if target is None:
            print(f"[Vespa BLE] Nenhum HockeyBot/Vespa encontrado no alcance.")
            return False

        print(f"[Vespa BLE] Conectando a {target.name} [{target.address}]...")
        self.client = BleakClient(target.address)
        await self.client.connect()
        self.is_connected = self.client.is_connected
        print(f"[Vespa BLE] Conectado com sucesso: {self.is_connected}")
        return self.is_connected

    async def send_wheel_speeds(self, v_left: float, v_right: float) -> bool:
        """
        Envia velocidades normalizadas [-1.0, 1.0] convertidas em PWM de 8 bits [-255, 255].
        """
        if not self.is_connected or self.client is None:
            return False

        pwm_l = int(max(-255, min(255, v_left * 255.0)))
        pwm_r = int(max(-255, min(255, v_right * 255.0)))

        # Pacote padrão: [HEADER: 0xAA, PWM_L (short), PWM_R (short), CHECKSUM]
        packet = struct.pack(">BhhB", 0xAA, pwm_l, pwm_r, (pwm_l + pwm_r) & 0xFF)
        try:
            await self.client.write_gatt_char(self.UART_RX_CHAR_UUID, packet, response=False)
            return True
        except Exception as e:
            print(f"[Vespa BLE] Erro no envio: {e}")
            return False

    async def disconnect(self) -> None:
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            self.is_connected = False
            print("[Vespa BLE] Desconectado.")
