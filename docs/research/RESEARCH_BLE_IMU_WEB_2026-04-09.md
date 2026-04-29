---
title: "BLE IMU Web Integration Research"
date: "2026-04-09"
status: active
---

# BLE IMU Web Integration Research

**Date:** 2026-04-09  
**Goal:** Feasibility of connecting Bluetooth IMU sensors to a web app for real-time figure skating analysis

---

## Executive Summary

BLE IMU integration is **technically feasible but architecturally constrained**. iOS Safari does not support Web Bluetooth — this makes pure browser-based (Option A) a non-starter for rink use where coaches use iPhones.

**Recommended:** Option B (native companion app, React Native) — full BLE access on iOS+Android, reuses React/TS expertise.  
**MVP starting point:** Option D (phone-as-IMU) — zero hardware cost, immediate deployment, modern phone IMUs are excellent.

---

## Key Constraints

| Browser | Web Bluetooth Support |
|---------|----------------------|
| Chrome (desktop/Android) | Yes |
| Edge | Yes |
| Firefox | No |
| Safari (macOS/iOS) | **No** — critical blocker |
| Chrome on iOS | No (WKWebView) |

**Bottom line:** Web Bluetooth works on Chrome/Edge desktop + Android only. iOS completely unsupported.

---

## Architecture Options

| Option | Approach | iOS | Rate | Verdict |
|--------|----------|-----|------|---------|
| A | Browser -> Web Bluetooth -> FastAPI | No | 50-100Hz | **Not viable** |
| **B** | **React Native -> BLE -> WebSocket -> FastAPI** | **Yes** | **100-200Hz** | **Recommended** |
| C | BLE Gateway (ESP32/RPi) -> WiFi | Yes | 100-200Hz | Viable but over-engineered |
| **D** | **Phone-as-IMU (zero hardware)** | **Yes** | **100-400Hz** | **Best MVP start** |

### Option B Details (Recommended)
- Libraries: `react-native-ble-plx` (v3.5.1, mature) or `react-native-ble-nitro` (v1.12.0, higher perf)
- Pros: Full BLE, background capable, offline cache, push notifications
- Cons: App store distribution needed

### Option D Details (MVP)
- Modern phones have 16-bit IMUs @ 100-400Hz with hardware sensor fusion
- React Native: `react-native-sensors` for full access
- Web fallback: `DeviceMotionEvent` (deprecated but supported)

---

## Data Format & Bandwidth

**9-axis IMU @ 100Hz:** 22 bytes/sample = 2.2 KB/s raw, ~48 kbps JSON, ~19 kbps binary (MessagePack).

**BLE throughput:** BLE 4.0 ~100kbps, BLE 4.2 ~250kbps, BLE 5.0 ~600kbps. 9-axis @ 200Hz easily fits.

**Packet loss:** Ice rink = low interference (<1%). Strategies: timestamps, sequence numbers, circular buffer.

---

## Phone-as-IMU Accuracy vs Dedicated IMU

| Metric | Phone (waist) | Dedicated IMU (foot) |
|--------|--------------|---------------------|
| Jump height | ±5cm | ±2cm |
| Rotation count | ±0.25 rev | ±0.1 rev |
| Takeoff time | ±10ms | ±5ms |
| Body lean | ±2° | N/A |

Phone placement: **waist belt (tight)** best overall.

---

## Recommended Rollout

### Phase 1: Phone-as-IMU MVP (Week 1-2)
- React Native companion app
- DeviceMotion API → WebSocket → FastAPI
- Jump detection, rotation counting, body lean
- Correlate IMU events with video frames

### Phase 2: BLE IMU Integration (Month 2-3)
- Add boot-mounted BLE IMUs (Movesense, ESP32)
- Multi-sensor fusion (Madgwick/Mahony)
- Blade angle detection from foot IMU

### Phase 3: Rink Gateway (Optional)
- ESP32-S3 gateway at rinkside
- Multi-skater simultaneous tracking

---

## Cost Estimate

| Component | Cost |
|-----------|------|
| Phase 1 (phone-as-IMU) | $0 |
| Phase 2 (2× Movesense + belt) | $320-420 |
| Phase 3 (ESP32-S3 gateway) | $60 |

---

## Key Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| iOS BLE permission denied | Low | Clear UX, guide through Settings |
| BLE packet loss during jumps | Medium | Buffer + timestamp reconstruction |
| IMU drift during spins | Medium | Complementary filter (accel correction) |
| Time sync drift (video vs IMU) | Medium | Audio clap sync, periodic resync |

---

## References

- **AIOnIce / Synergie:** `github.com/Mart1t1/Synergie` — Xsens IMU + FastAPI for skating jump classification
- **react-native-ble-plx:** `npmjs.com/package/react-native-ble-plx` (v3.5.1)
- **Bleak (Python BLE):** `github.com/hbldh/bleak` (2,374 stars)
- **Movesense React Native:** `github.com/dyarfaradj/Movesense-Bluetooth-Sensor`
- **Web Bluetooth API:** developer.mozilla.org/docs/Web/API/Web_Bluetooth_API

**Full 641-line analysis preserved in git history.**
