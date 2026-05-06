# EdgeSense Capture — Camera & Sensor Enhancement Design

## Goals

1. **Real-time IMU metrics** — gyroscope, accelerometer, edge angle / velocity on dedicated screen
2. **Battery level** — read WT901 supply voltage, display in BLE list and camera overlay
3. **Professional camera UI** — grid overlay, orientation lock, settings panel (FPS, resolution)
4. **Sensor configuration** — change return rate (Hz), rename Bluetooth device name
5. **Export & share** — save to `Downloads/EdgeSense/`, then native share of ZIP archive

## Architecture

### New Files

| File | Responsibility |
|---|---|
| `lib/ble/wt901_commander.dart` | Send WitMotion protocol commands: unlock, set return rate, read battery, rename |
| `lib/ble/battery_service.dart` | Periodic battery polling for connected IMU devices |
| `lib/metrics/metrics_screen.dart` | Real-time gyro/accel/angle dashboard with animated gauges |
| `lib/camera/grid_overlay.dart` | Rule-of-thirds grid compositing overlay |
| `lib/camera/camera_settings_sheet.dart` | Bottom sheet: resolution, FPS, orientation lock |
| `lib/sensor/sensor_settings_sheet.dart` | Bottom sheet: return rate, rename device |

### Modified Files

| File | Changes |
|---|---|
| `lib/ble/ble_manager.dart` | Add `readBattery()`, `setReturnRate()`, `renameDevice()`, expose `batteryLevels` map |
| `lib/ble/ble_scan_screen.dart` | Show battery chip per device, sensor settings button |
| `lib/ble/imudevice.dart` | Split from `ble_manager.dart` — cleaner separation |
| `lib/camera/camera_ready_screen.dart` | Add grid toggle, orientation lock, settings FAB, battery overlay |
| `lib/camera/recorder.dart` | Support resolution/fps change, orientation lock |
| `lib/export/exporter.dart` | Save to Downloads, trigger `share_plus` |
| `lib/main.dart` | Add route for metrics screen |
| `pubspec.yaml` | Add `share_plus`, `intl` dependencies |

## WitMotion Protocol Commands

Unlock sequence (required before any config change):
```
0xFF 0xAA 0x69 0x88 0xB5  // unlock
<config command>
0xFF 0xAA 0x00 0x00 0x00  // save
```

Read battery voltage (register 0x5C):
```
0xFF 0xAA 0x27 0x5C 0x00  // read register 0x5C
// Response: 0x55 0x5C BATVAL_L BATVAL_H ...
// Voltage = BATVAL / 100.0  (e.g. 420 = 4.20V)
```

Set return rate (register 0x03):
```
0xFF 0xAA 0x03 <rate> 0x00  // rate: 0x01=0.2Hz, 0x06=10Hz, 0x09=100Hz, 0x0B=200Hz
```

Rename device (register 0x75 device address):
```
0xFF 0xAA 0x75 <new_id> 0x00  // set device address / broadcast name suffix
```

## Battery Voltage Display

- **BLE scan screen**: small battery icon + percentage per assigned device tile
- **Camera overlay**: top-right battery indicator (left/right)
- **Color coding**: green (>3.7V), orange (3.5-3.7V), red (<3.5V)

## Camera UI Design

```
+----------------------------------+
| [grid]  [settings]      [ battery] |
|                                  |
|         CameraPreview            |
|         + grid overlay           |
|                                  |
|  [flip]  [REC ●]  [metrics]      |
+----------------------------------+
```

- **Top bar**: grid toggle, settings gear, battery levels
- **Center**: CameraPreview with optional grid (rule of thirds)
- **Bottom**: flip camera, record button, metrics button
- **Orientation**: support portrait, landscapeLeft, landscapeRight, portraitUpsideDown

## Sensor Metrics Screen

Three-column layout:
- **Left column**: Gyroscope X/Y/Z (radial gauges, °/s)
- **Center**: Accelerometer X/Y/Z (bars, g)
- **Right column**: Edge angle (roll from quaternion, °) + velocity estimate

Values update at sensor return rate (up to 200Hz, UI throttled to 30fps).

## Export Flow

1. After stop recording, `Exporter.export()` saves:
   - Video: `Downloads/EdgeSense/capture_YYYYMMDD_HHMMSS.mp4`
   - IMU CSV: `capture_YYYYMMDD_HHMMSS_left.csv`, `..._right.csv`
   - Manifest JSON: `capture_YYYYMMDD_HHMMSS.json`
2. ZIP all three: `capture_YYYYMMDD_HHMMSS.zip`
3. Trigger `Share.shareXFiles([zip])` for native share sheet

## Error Handling

- Battery read fails silently (shows "—"), retries every 30s
- Rename fails → show Snackbar "Не удалось переименовать"
- Return rate change → requires disconnect/reconnect or takes effect immediately per docs
- Camera settings change → requires re-initialization of CameraController
