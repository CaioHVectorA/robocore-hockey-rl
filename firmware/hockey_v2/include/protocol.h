#pragma once
#include <stdint.h>
#include <stddef.h>

namespace hockey {
struct Command { uint16_t sequence; int16_t left; int16_t right; uint16_t ttl; };
inline uint16_t read16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
inline uint16_t crc16(const uint8_t* data, size_t n) {
  uint16_t crc = 0xffff;
  for (size_t i = 0; i < n; ++i) {
    crc ^= uint16_t(data[i]) << 8;
    for (int j = 0; j < 8; ++j) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1;
  }
  return crc;
}
inline bool decode(const uint8_t* p, size_t size, Command& cmd) {
  if (size != 13 || p[0] != 0xaa || p[1] != 0x55 || p[2] != 2 || crc16(p,11) != read16(p+11)) return false;
  cmd = {read16(p+3), int16_t(read16(p+5)), int16_t(read16(p+7)), read16(p+9)};
  return cmd.left >= -255 && cmd.left <= 255 && cmd.right >= -255 && cmd.right <= 255 && cmd.ttl >= 20 && cmd.ttl <= 200;
}
inline bool newer(uint16_t candidate, uint16_t previous) { return int16_t(candidate-previous) > 0; }
}
