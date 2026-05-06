# IMU Capture & Sensor Fusion Design

## Overview

Offline-first Flutter capture app для сбора IMU (WT901BLE) + видео с телефона, синхронизация по unified `t0`, ручной экспорт `.esense.zip` → загрузка через сайт → backend sensor fusion (MogaNet-B + IMU augmentation → 3D скелет).

**Key principle:** минимум движущихся частей. Нет real-time upload, нет clap-detect. `Record` одна кнопка → все источники пишут с `relative_timestamp_ms = now - t0`.

## Goals

1. Flutter MVP: скан 2 BLE сенсоров, unified t0 record, real-time overlay (debug), calibration, offline export.
2. `.esense.zip` — стандартный пакет для обработки.
3. Backend: time-aligned sensor fusion (video pose + IMU → 3D skeleton).
4. EdgeAngle + airtime + landing force из IMU, augment pose ankle joints.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│           FLUTTER CAPTURE APP (Offline)            │
│  ┌──────────────┐      ┌──────────────┐          │
│  │  WT901 Left  │◄────►│  BLE Manager │          │
│  └──────────────┘      │  (bleak via  │          │
│  ┌──────────────┐      │   platform   │          │
│  │  WT901 Right │◄────►│   channels)  │          │
│  └──────────────┘      └──────┬───────┘          │
│                               │                    │
│  ┌────────────────────────────▼──────────────────┐│
│  │           Capture Controller (t0 trigger)      ││
│  │  - starts Camera API + BLE streams simultaneously│
│  │  - writes relative_timestamp_ms for every sample │
│  │  - provides real-time overlay (debug)           │
│  └────────────────────────────┬──────────────────┘│
│                               │                    │
│  ┌──────────────┐    ┌────────┴────────┐         │
│  │ Calibration  │    │   Review &       │         │
│  │ (quat_ref)   │    │   Export         │         │
│  └──────────────┘    │   (.esense.zip)  │         │
│                      └────────┬────────┘         │
└───────────────────────────────┼────────────────────┘
                              │ .esense.zip
┌───────────────────────────────┼────────────────────┐
│           WEB / BACKEND         ▼                    │
│  ┌──────────────┐      ┌──────────────┐          │
│  │  FastAPI     │◄────►│  arq worker  │          │
│  │  upload      │      │  (sensor     │          │
│  │  endpoint    │      │   fusion)    │          │
│  └──────────────┘      └──────┬───────┘          │
│                               │                    │
│  ┌────────────────────────────▼──────────────────┐│
│  │           Sensor Fusion Pipeline               ││
│  │  1. Video → MogaNet-B → 2D/3D pose (H36M)     ││
│  │  2. IMU → edge angle (quat vs calib)          ││
│  │  3. IMU → force profile (acc_norm), airtime   ││
│  │  4. Augment: apply IMU ankle orientation      ││
│  │     to pose 3D joints                         ││
│  │  5. Report: JSON metrics + visualization      ││
│  └───────────────────────────────────────────────┘│
└────────────────────────────────────────────────────┘
```

## Data Format: `.esense.zip`

ZIP-архив содержит:

| File | Format | Description |
|------|--------|-------------|
| `manifest.json` | JSON | metadata, timestamps, settings |
| `video.mp4` | H.264 | Camera API output, 60fps |
| `left.imu` | Protobuf | raw IMU stream, left skate |
| `right.imu` | Protobuf | raw IMU stream, right skate |
| `calibration.json` | JSON | quat_ref for both sensors |

### `manifest.json`

```json
{
  "version": "1.0",
  "created_at": "2026-05-06T14:30:00Z",
  "t0_ms": 1714998600000,
  "duration_ms": 45230,
  "video": {
    "filename": "video.mp4",
    "fps": 60,
    "width": 1920,
    "height": 1080,
    "start_offset_ms": 0
  },
  "imu": {
    "left": {
      "filename": "left.imu",
      "sensor_id": "WT901_12345",
      "sample_rate_hz": 100,
      "start_offset_ms": 0
    },
    "right": {
      "filename": "right.imu",
      "sensor_id": "WT901_67890",
      "sample_rate_hz": 100,
      "start_offset_ms": 0
    }
  },
  "calibration": {
    "filename": "calibration.json"
  }
}
```

### Protobuf schema (`imu.proto`)

```protobuf
syntax = "proto3";

message IMUSample {
  uint64 relative_timestamp_ms = 1;  // ms from t0
  float acc_x = 2;
  float acc_y = 3;
  float acc_z = 4;
  float gyro_x = 5;
  float gyro_y = 6;
  float gyro_z = 7;
  float quat_w = 8;  // q0
  float quat_x = 9;  // q1
  float quat_y = 10; // q2
  float quat_z = 11; // q3
}

message IMUStream {
  repeated IMUSample samples = 1;
}
```

Flutter app stores as **length-delimited** messages (varint prefix per message) for streaming read. Backend decoder reads length-delimited.

### `calibration.json`

```json
{
  "left": {
    "quat_ref": [0.98, 0.0, 0.17, 0.0],
    "calibrated_at": "2026-05-06T14:25:00Z"
  },
  "right": {
    "quat_ref": [0.97, 0.0, 0.23, 0.0],
    "calibrated_at": "2026-05-06T14:25:00Z"
  }
}
```

## Flutter App Modules

### 1. BLE Manager

**Library:** `flutter_blue_plus` (Bluetooth LE).

**Scan & Connect:**
- Service UUID filter: WT901 advertises with known service UUID (discover empirically).
- Connect 2 devices by MAC/name (left/right assignment).

**Protocol Parsing (WT901 0x55):**
- Packet: 11 bytes, starts with `0x55`.
- Byte 1: type (`0x51` acc, `0x52` gyro, `0x53` angle, `0x59` quaternion).
- After connect: configure output content to send ONLY `0x51` (acc) and `0x59` (quat) via WitMotion config characteristic.
- Parse payload, store with `timestamp = SystemClock.elapsedRealtime() - t0`.

**Settings (exposed in UI):**
- `sample_rate_hz`: 10, 20, 50, 100, 200
- `bandwidth_hz`: 98, 196, 392 (default 98)
- `axis_mode`: 6-axis (hard fusion, no mag) or 9-axis (default: 6-axis for ice)
- `output_content`: bitmask for packet types (default: acc + quat)

### 2. Camera Recorder

**Library:** `camera` (Flutter official).

- Initialize with 60fps, 1080p.
- On record start: `video_t0 = t0` (same as BLE).
- Metadata: `start_offset_ms = 0` (aligned by t0).

### 3. Capture Controller (t0 trigger)

```dart
class CaptureController {
  DateTime? t0;
  List<IMUSample> leftBuffer = [];
  List<IMUSample> rightBuffer = [];

  Future<void> start() async {
    t0 = DateTime.now();
    await Future.wait([
      cameraController.startVideoRecording(),
      bleManager.startStream(leftDevice, t0),
      bleManager.startStream(rightDevice, t0),
    ]);
  }

  Future<void> stop() async {
    await Future.wait([
      cameraController.stopVideoRecording(),
      bleManager.stopStream(leftDevice),
      bleManager.stopStream(rightDevice),
    ]);
  }
}
```

Every sample gets `relative_timestamp_ms = packetReceiveTime - t0.millisecondsSinceEpoch`.

### 4. Real-time Overlay (Debug)

During recording (or pre-record preview):
- Show current `EdgeAngle` from latest quat sample.
- Computed as: `edge_angle = atan2(2*(q0*q2 + q1*q3), 1 - 2*(q2^2 + q3^2))` (roll from quaternion) minus calibration roll.
- Display: left/right edge angle in degrees, small gauge.
- Latency: acceptable ~50-100ms for debug only.

### 5. Calibration

**Flow:**
1. User stands still on ice, skates parallel.
2. Tap "Calibrate".
3. App captures 1 second of quat samples (100 samples).
4. Compute mean quaternion = `quat_ref`.
5. Save in app memory (used for overlay + export).

**Algorithm:**
- Average quaternions via eigen-decomposition of sum of outer products.
- Simpler: normalize mean of qx, qy, qz, qw.

### 6. Review & Export

**Review screen:**
- Video playback with IMU graph overlay (AccTotal, EdgeAngle) synchronized.
- Slider to seek, graph auto-scrolls.

**Export:**
- Write manifest, protobuf files, calibration JSON, video to temp dir.
- Zip to `.esense.zip`.
- Share sheet / save to downloads.

## Backend Sensor Fusion

### Input

`.esense.zip` uploaded via FastAPI → saved to R2 → arq task queued.

### Processing Pipeline (new `ml/src/sensor_fusion/`)

#### 1. Decode (`decoder.py`)

- Unzip, parse manifest, load protobuf streams into `pandas.DataFrame`.
- Columns: `timestamp_ms, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, quat_w, quat_x, quat_y, quat_z`.

#### 2. Preprocess (`preprocess.py`)

- **Butterworth low-pass filter** on acc (fc=20Hz) and gyro (fc=10Hz) to remove blade vibration.
- **Compute derived signals:**
  - `acc_norm = sqrt(acc_x^2 + acc_y^2 + acc_z^2)`
  - `edge_angle_deg` from quat relative to calibration

#### 3. Video Pose (`video_pose.py`)

- Run existing MogaNet-B pipeline on `video.mp4`.
- Output: H36M 17kp per frame, 60fps.
- Interpolate to IMU timestamps (100Hz) via linear interpolation for ankle keypoints.

#### 4. Sensor Fusion (`fusion.py`)

**Goal:** augment ankle joints with IMU orientation.

- IMU is on heel → corresponds to H36M ankle joint (index depends on model).
- For each timestamp:
  - Get 3D ankle position from pose (global coords).
  - Get IMU quat (local→global).
  - Apply IMU rotation to local foot vector (e.g., [0, 0, -1] for blade direction) → global foot orientation.
  - Use to orient ankle joint, derive knee-ankle-blade angle.

**Airtime detection:**
- Find windows where `acc_norm < 1.2g` (±threshold) for >100ms.
- Compute: `airtime_ms`, `takeoff_acc_peak`, `landing_acc_peak`.

**Landing force:**
- Peak `acc_norm` in first 200ms after airtime end.

**Edge stability:**
- Std dev of `edge_angle_deg` during glide segments (no airtime).

#### 5. Output Report (`report.py`)

JSON:

```json
{
  "video_id": "...",
  "duration_ms": 45230,
  "elements": [
    {
      "type": "jump",
      "start_ms": 1234,
      "end_ms": 2567,
      "airtime_ms": 420,
      "landing_force_g": 4.2,
      "takeoff_edge": "outer",
      "landing_edge": "inner"
    }
  ],
  "metrics": {
    "avg_edge_angle_left": -12.3,
    "avg_edge_angle_right": 8.7,
    "landing_stability": 0.85
  },
  "imu_enhanced_pose": "pose_imu_3d.npz"
}
```

## Testing Strategy

| Layer | Test |
|-------|------|
| Flutter BLE | Mock BLE characteristic, verify parsing of `0x55 0x51` and `0x55 0x59` packets |
| Flutter Sync | Mock camera + BLE streams with known t0, verify all samples start at ~0ms |
| Protobuf | Roundtrip: encode → decode, verify field values |
| Backend decoder | Load sample `.esense.zip`, assert DataFrame shape and column names |
| Backend fusion | Known IMU orientation → expected edge angle. Synthetic airtime signal → correct detection |
| End-to-end | Full pipeline: sample zip → report JSON with expected metrics |

## File Structure

```
flutter_app/
├── lib/
│   ├── ble/
│   │   ├── wt901_parser.dart
│   │   └── ble_manager.dart
│   ├── camera/
│   │   └── recorder.dart
│   ├── capture/
│   │   └── controller.dart
│   ├── calibration/
│   │   └── calibration_service.dart
│   ├── export/
│   │   ├── protobuf_gen/
│   │   │   └── imu.pb.dart (generated)
│   │   └── exporter.dart
│   └── main.dart
├── proto/
│   └── imu.proto
└── pubspec.yaml

ml/src/sensor_fusion/
├── __init__.py
├── decoder.py
├── preprocess.py
├── video_pose.py
├── fusion.py
└── report.py
```

## Open Questions / Future Work

1. **IMU→3D augmentation:** exact biomechanical model of foot (blade direction vector in IMU frame) needs calibration on ice. Estimate empirically.
2. **Flutter BLE reliability:** iOS background BLE may drop connection. Test on both platforms.
3. **Video+IMU interpolation:** pose at 60fps, IMU at 100Hz. Upsample pose or downsample IMU? Upsample pose with linear interp.
4. **Quaternion averaging:** calibration uses simple mean — should use proper quaternion averaging for accuracy. OK for MVP.
5. **Multiple camera angles:** current design single camera. Multi-cam → extend manifest with array.
6. **Real-time upload:** out of MVP scope. Future: WebSocket/MQTT from ReTerminal.

## Dependencies

### Flutter
- `flutter_blue_plus: ^1.32.0`
- `camera: ^0.10.5`
- `protobuf: ^3.1.0`
- `archive: ^3.4.0` (zip)
- `path_provider: ^2.1.0`

### Python (backend)
- `protobuf` (already in pyproject)
- `scipy` (already in pyproject, for Butterworth)
- `pandas` (already in pyproject)
- Existing: `ml/src/pose_estimation/`, `ml/src/analysis/`

## MVP Scope Checklist

- [ ] Flutter: scan & connect 2 WT901
- [ ] Flutter: unified t0 record (video + IMU)
- [ ] Flutter: real-time edge angle overlay
- [ ] Flutter: calibration (quat_ref)
- [ ] Flutter: export `.esense.zip`
- [ ] Backend: `/api/v1/capture/upload` endpoint
- [ ] Backend: arq worker decodes `.esense.zip`
- [ ] Backend: Butterworth filter + edge angle compute
- [ ] Backend: airtime detection
- [ ] Backend: augment pose with IMU orientation
- [ ] Backend: JSON report
