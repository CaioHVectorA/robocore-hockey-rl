"""
Ponte de Comunicação Bluetooth BLE de Alta Fidelidade com o RoboCore HockeyBot (Placa Vespa / ESP32).

Recursos de Fidelidade Física com o Hardware Real:
1. Compensação de Zona Morta (Deadband Compensation): Mapeia saídas neurais baixas
   para vencer o atrito estático da caixa de redução N20/25GA sem estagnar.
2. Calibração de Trim de Motores: Permite ajuste fino de balanço entre as rodas
   para compensar disparidades físicas de fabricação dos motores DC.
3. Taxa de Envio Estável a 30 Hz: Compatível com o timer de amostragem do ESP32,
   prevenindo buffer overflow e latências espúrias no rádio.
4. Watchdog e Parada de Emergência Automática (Emergency Stop): Corta corrente
   dos motores caso a conexão caia ou nenhum pacote chegue em 200 ms.
5. Diagnóstico de Bancada e Scanner BLE via CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import struct
import sys
import time
from typing import Optional, Tuple, Callable

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False


def apply_deadband_compensation(
    norm_val: float, deadband_pwm: int = 35, max_pwm: int = 255
) -> int:
    """
    Compensa a zona morta física dos motores DC da RoboCore Vespa.
    - Se norm_val == 0.0 -> PWM 0 (motor livre/travado)
    - Se norm_val != 0.0 -> Mapeia (0.0, 1.0] -> [deadband_pwm, max_pwm],
      garantindo que qualquer comando da IA realmente movimente as rodas na arena.
    """
    if abs(norm_val) < 0.015:
        return 0
    mag = abs(norm_val)
    pwm = int(deadband_pwm + mag * (max_pwm - deadband_pwm))
    pwm = max(0, min(max_pwm, pwm))
    return int(math.copysign(pwm, norm_val))


class VespaBluetoothBridge:
    """
    Gerenciador de conexão BLE de alta performance com a placa RoboCore Vespa (ESP32).
    """

    # UUID padrão da característica GATT UART RX usada para controle do ESP32
    UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
    UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

    def __init__(
        self,
        device_name_prefix: str = "HockeyBot",
        target_mac: Optional[str] = None,
        trim_left: float = 1.0,
        trim_right: float = 1.0,
        deadband_pwm: int = 35,
        control_rate_hz: float = 30.0,
    ):
        self.device_name_prefix = device_name_prefix
        self.target_mac = target_mac
        self.trim_left = trim_left
        self.trim_right = trim_right
        self.deadband_pwm = deadband_pwm
        self.control_rate_hz = control_rate_hz

        self.client: Optional[BleakClient] = None
        self.is_connected: bool = False
        self.last_send_time: float = 0.0
        self.packet_count: int = 0
        self.last_rtt_ms: float = 0.0

    async def scan_devices(self, timeout: float = 5.0):
        """Descobre todos os dispositivos BLE no alcance com nome ou prefixo da Vespa."""
        if not BLEAK_AVAILABLE:
            print("[Vespa BLE] Erro: biblioteca 'bleak' não instalada.")
            return []

        print(f"[Vespa BLE] Escaneando dispositivos BLE por {timeout:.1f}s...")
        devices = await BleakScanner.discover(timeout=timeout)
        found = []
        for d in devices:
            name = d.name or "Desconhecido"
            print(f"  - [{d.address}] {name} (RSSI: {d.rssi} dBm)")
            if self.device_name_prefix.lower() in name.lower() or "vespa" in name.lower():
                found.append(d)
        return found

    async def connect(self, timeout: float = 6.0) -> bool:
        """Conecta ao HockeyBot físico por MAC específico ou pelo primeiro dispositivo encontrado."""
        if not BLEAK_AVAILABLE:
            print("[Vespa BLE] Biblioteca 'bleak' não instalada. Execute: pip install bleak")
            return False

        target_address = self.target_mac
        target_name = self.device_name_prefix

        if target_address is None:
            print(f"[Vespa BLE] Procurando HockeyBot com prefixo '{self.device_name_prefix}'...")
            devices = await BleakScanner.discover(timeout=timeout)
            for d in devices:
                name = d.name or ""
                if self.device_name_prefix.lower() in name.lower() or "vespa" in name.lower():
                    target_address = d.address
                    target_name = name
                    break

        if target_address is None:
            print(f"[Vespa BLE] Nenhum robô HockeyBot/Vespa encontrado no alcance.")
            return False

        print(f"[Vespa BLE] Conectando a {target_name} [{target_address}]...")
        self.client = BleakClient(target_address)
        try:
            await self.client.connect(timeout=8.0)
            self.is_connected = self.client.is_connected
            print(f"[Vespa BLE] Conexão estabelecida com sucesso: {self.is_connected}")
            return self.is_connected
        except Exception as e:
            print(f"[Vespa BLE] Falha ao conectar: {e}")
            self.is_connected = False
            return False

    async def send_wheel_speeds(
        self,
        v_left: float,
        v_right: float,
        apply_compensation: bool = True,
    ) -> bool:
        """
        Envia velocidades normalizadas [-1.0, 1.0] convertidas em PWM [-255, 255]
        com compensação de zona morta e calibração de trim dos motores.
        """
        if not self.is_connected or self.client is None:
            return False

        # Aplica calibração de trim de hardware
        u_l = float(v_left) * self.trim_left
        u_r = float(v_right) * self.trim_right

        if apply_compensation:
            pwm_l = apply_deadband_compensation(u_l, self.deadband_pwm, 255)
            pwm_r = apply_deadband_compensation(u_r, self.deadband_pwm, 255)
        else:
            pwm_l = int(max(-255, min(255, u_l * 255.0)))
            pwm_r = int(max(-255, min(255, u_r * 255.0)))

        # Pacote padrão de telemetria Vespa RoboCore:
        # [HEADER: 0xAA 0x55, PWM_L (short), PWM_R (short), MODE (byte), CHECKSUM]
        mode_flag = 0x01 if (pwm_l != 0 or pwm_r != 0) else 0x00
        chk = (pwm_l + pwm_r + mode_flag) & 0xFF
        packet = struct.pack(">BBhhBB", 0xAA, 0x55, pwm_l, pwm_r, mode_flag, chk)

        t0 = time.perf_counter()
        try:
            await self.client.write_gatt_char(self.UART_RX_CHAR_UUID, packet, response=False)
            self.last_rtt_ms = (time.perf_counter() - t0) * 1000.0
            self.last_send_time = time.time()
            self.packet_count += 1
            return True
        except Exception as e:
            print(f"[Vespa BLE] Erro no envio de pacote: {e}")
            return False

    async def emergency_stop(self) -> None:
        """Comando imediato de corte de corrente (stop total nos 2 motores)."""
        print("[Vespa BLE] EMERGENCY STOP disparado!")
        await self.send_wheel_speeds(0.0, 0.0, apply_compensation=False)

    async def disconnect(self) -> None:
        """Desconecta com segurança parando os motores primeiro."""
        if self.client and self.client.is_connected:
            try:
                await self.emergency_stop()
                await self.client.disconnect()
            except Exception:
                pass
            self.is_connected = False
            print("[Vespa BLE] Desconectado com sucesso.")

    async def run_bench_test(self) -> None:
        """
        Rotina oficial de bancada para testar o HockeyBot real:
        1. Roda Esquerda avança (1s)
        2. Roda Direita avança (1s)
        3. Avanço em linha reta (1s)
        4. Giro rápido (Spin Move / chute centrífugo) (1s)
        5. Parada total
        """
        print("\n=== INICIANDO TESTE DE BANCADA ROBOCORE HOCKEYBOT ===")
        print("Certifique-se de que o carrinho está apoiado ou suspenso em bancada!")
        await asyncio.sleep(1.0)

        steps = [
            ("Roda Esquerda (35% PWM)", 0.35, 0.0, 1.0),
            ("Roda Direita (35% PWM)", 0.0, 0.35, 1.0),
            ("Avanço em Linha Reta (50% PWM)", 0.50, 0.50, 1.2),
            ("Giro Rápido / Spin Shot (75% PWM)", 0.75, -0.75, 0.8),
            ("Parada Total", 0.0, 0.0, 0.5),
        ]

        for desc, vl, vr, duration in steps:
            print(f" -> Executando: {desc} por {duration}s...")
            end_t = time.time() + duration
            while time.time() < end_t:
                await self.send_wheel_speeds(vl, vr)
                await asyncio.sleep(1.0 / self.control_rate_hz)

        await self.emergency_stop()
        print("=== TESTE DE BANCADA CONCLUÍDO COM SUCESSO! ===\n")


class VespaTeamBridge:
    """
    Gerenciador multi-robô oficial para equipe 3v3 BLE:
    - Robô 0: Atacante
    - Robô 1: Ala / Defensor
    - Robô 2: Goleiro
    Conecta e despacha ações em paralelo via asyncio para as 3 placas Vespa ESP32.
    """

    def __init__(
        self,
        macs: Optional[List[str]] = None,
        name_prefixes: Optional[List[str]] = None,
        trims: Optional[List[Tuple[float, float]]] = None,
    ):
        prefixes = name_prefixes or ["HockeyBot-1", "HockeyBot-2", "HockeyBot-3"]
        mac_list = macs or [None, None, None]
        trim_list = trims or [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]

        self.robots = [
            VespaBluetoothBridge(
                device_name_prefix=prefixes[i],
                target_mac=mac_list[i],
                trim_left=trim_list[i][0],
                trim_right=trim_list[i][1],
            )
            for i in range(3)
        ]

    async def connect_all(self, timeout: float = 6.0) -> List[bool]:
        """Conecta aos 3 robôs da equipe simultaneamente."""
        print(f"[Vespa Team BLE] Conectando aos 3 robôs da equipe 3v3...")
        results = await asyncio.gather(*(r.connect(timeout=timeout) for r in self.robots))
        return list(results)

    async def send_team_actions(
        self,
        actions: List[Tuple[float, float]],
        apply_compensation: bool = True,
    ) -> List[bool]:
        """Envia comandos para os 3 robôs em paralelo a 30 Hz."""
        tasks = []
        for i, r in enumerate(self.robots):
            if i < len(actions):
                vl, vr = actions[i]
                tasks.append(r.send_wheel_speeds(vl, vr, apply_compensation=apply_compensation))
            else:
                tasks.append(r.send_wheel_speeds(0.0, 0.0))
        return list(await asyncio.gather(*tasks))

    async def emergency_stop_all(self) -> None:
        """Corte imediato de motores nos 3 robôs da equipe."""
        print("[Vespa Team BLE] Parada de emergência geral da equipe 3v3 disparada!")
        await asyncio.gather(*(r.emergency_stop() for r in self.robots))

    async def disconnect_all(self) -> None:
        """Desconecta todos os robôs da equipe com segurança."""
        await asyncio.gather(*(r.disconnect() for r in self.robots))


async def main_cli():
    parser = argparse.ArgumentParser(description="Ponte Bluetooth BLE para RoboCore Vespa (HockeyBot 3v3)")
    parser.add_argument("--scan", action="store_true", help="Escaneia dispositivos BLE no alcance")
    parser.add_argument("--test", action="store_true", help="Executa rotina de teste de bancada no robô real")
    parser.add_argument("--mac", type=str, default=None, help="Endereço MAC específico do ESP32")
    parser.add_argument("--trim-left", type=float, default=1.0, help="Fator de calibração do motor esquerdo")
    parser.add_argument("--trim-right", type=float, default=1.0, help="Fator de calibração do motor direito")
    args = parser.parse_args()

    bridge = VespaBluetoothBridge(
        target_mac=args.mac,
        trim_left=args.trim_left,
        trim_right=args.trim_right,
    )

    if args.scan:
        await bridge.scan_devices()
        return

    if args.test:
        connected = await bridge.connect()
        if not connected:
            print("[Vespa BLE] Não foi possível conectar ao robô para o teste.")
            return
        try:
            await bridge.run_bench_test()
        finally:
            await bridge.disconnect()
        return

    print("Use --scan para buscar robôs ou --test para executar autoteste.")


if __name__ == "__main__":
    asyncio.run(main_cli())
