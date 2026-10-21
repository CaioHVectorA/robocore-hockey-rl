import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from bridge.vespa_ble_bridge import encode_command


@pytest.fixture(scope="module")
def decoder(tmp_path_factory):
    if not shutil.which("g++"):
        pytest.skip("Native C++ compiler not installed")
    folder = tmp_path_factory.mktemp("firmware-protocol")
    source = folder / "check.cpp"
    source.write_text("""#include <iostream>
#include "protocol.h"
int main(){
  unsigned char p[13];std::cin.read((char*)p,13);hockey::Command c;
  if(!hockey::decode(p,std::cin.gcount(),c))return 1;
  if(!hockey::newer(0,65535)||hockey::newer(65535,0))return 2;
  std::cout<<c.sequence<<" "<<c.left<<" "<<c.right<<" "<<c.ttl;
}
""")
    binary = folder / "decode"
    include = Path(__file__).resolve().parents[1] / "firmware/hockey_v2/include"
    subprocess.run(
        ["g++", "-std=c++11", "-Wall", "-Werror", "-I", str(include), str(source), "-o", str(binary)],
        check=True,
    )
    return binary


def test_python_packets_match_actual_firmware_decoder(decoder):
    rng = np.random.default_rng(12)
    for _ in range(30):
        seq = int(rng.integers(65536))
        left, right = (int(x) for x in rng.integers(-255, 256, 2))
        ttl = int(rng.integers(20, 201))
        result = subprocess.run(
            [str(decoder)], input=encode_command(seq, left, right, ttl), capture_output=True, check=True
        )
        assert tuple(map(int, result.stdout.split())) == (seq, left, right, ttl)


def test_firmware_rejects_corrupt_packet(decoder):
    packet = bytearray(encode_command(5, 100, -100))
    packet[8] ^= 128
    assert subprocess.run([str(decoder)], input=packet, capture_output=True).returncode == 1
