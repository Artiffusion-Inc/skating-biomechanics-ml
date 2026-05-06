import 'package:flutter_test/flutter_test.dart';
import '../../lib/i18n/strings.g.dart';

void main() {
  group('Russian translations', () {
    late TranslationsRu t;

    setUpAll(() {
      t = TranslationsRu();
    });

    test('app namespace keys are non-empty', () {
      expect(t.app.title, isNotEmpty);
    });

    test('permissions namespace keys are non-empty', () {
      expect(t.permissions.grant, isNotEmpty);
      expect(t.permissions.bleRequired, isNotEmpty);
      expect(t.permissions.required, isNotEmpty);
      expect(t.permissions.list, isNotEmpty);
    });

    test('ble namespace keys are non-empty', () {
      expect(t.ble.scanTitle, isNotEmpty);
      expect(t.ble.scan, isNotEmpty);
      expect(t.ble.rescan, isNotEmpty);
      expect(t.ble.next, isNotEmpty);
      expect(t.ble.bluetoothOff, isNotEmpty);
      expect(t.ble.assignHint, isNotEmpty);
      expect(t.ble.left, isNotEmpty);
      expect(t.ble.right, isNotEmpty);
      expect(t.ble.bothConnected, isNotEmpty);
      expect(t.ble.oneConnected, isNotEmpty);
      expect(t.ble.connecting, isNotEmpty);
      expect(t.ble.sensorSettings, isNotEmpty);
      expect(t.ble.status.connected, isNotEmpty);
      expect(t.ble.status.disconnected, isNotEmpty);
      expect(t.ble.errors.locationRequired, isNotEmpty);
      expect(t.ble.battery.title, isNotEmpty);
      expect(t.ble.battery.unknown, isNotEmpty);
      expect(t.ble.battery.request, isNotEmpty);
      expect(t.ble.battery.unit, isNotEmpty);
      expect(t.ble.returnRate.title, isNotEmpty);
      expect(t.ble.returnRate.hint, isNotEmpty);
      expect(t.ble.returnRate.select, isNotEmpty);
      expect(t.ble.returnRate.hz02, isNotEmpty);
      expect(t.ble.returnRate.hz05, isNotEmpty);
      expect(t.ble.returnRate.hz1, isNotEmpty);
      expect(t.ble.returnRate.hz2, isNotEmpty);
      expect(t.ble.returnRate.hz5, isNotEmpty);
      expect(t.ble.returnRate.hz10, isNotEmpty);
      expect(t.ble.returnRate.hz20, isNotEmpty);
      expect(t.ble.returnRate.hz50, isNotEmpty);
      expect(t.ble.returnRate.hz100, isNotEmpty);
      expect(t.ble.returnRate.hz200, isNotEmpty);
      expect(t.ble.rename.title, isNotEmpty);
      expect(t.ble.rename.action, isNotEmpty);
      expect(t.ble.rename.dialogTitle, isNotEmpty);
      expect(t.ble.rename.label, isNotEmpty);
      expect(t.ble.rename.placeholder, isNotEmpty);
      expect(t.ble.rename.cancel, isNotEmpty);
      expect(t.ble.rename.save, isNotEmpty);
    });

    test('calibration namespace keys are non-empty', () {
      expect(t.calibration.title, isNotEmpty);
      expect(t.calibration.instruction, isNotEmpty);
      expect(t.calibration.left, isNotEmpty);
      expect(t.calibration.right, isNotEmpty);
      expect(t.calibration.start, isNotEmpty);
      expect(t.calibration.calibrate, isNotEmpty);
      expect(t.calibration.skip, isNotEmpty);
      expect(t.calibration.done, isNotEmpty);
      expect(t.calibration.running, isNotEmpty);
      expect(t.calibration.errorPrefix, isNotEmpty);
      expect(t.calibration.noData, isNotEmpty);
      expect(t.calibration.startCapture, isNotEmpty);
      expect(t.calibration.seconds, isNotEmpty);
    });

    test('camera namespace keys are non-empty', () {
      expect(t.camera.title, isNotEmpty);
      expect(t.camera.startCapture, isNotEmpty);
      expect(t.camera.unavailable, isNotEmpty);
      expect(t.camera.retry, isNotEmpty);
      expect(t.camera.resolution, isNotEmpty);
      expect(t.camera.orientation, isNotEmpty);
      expect(t.camera.portrait, isNotEmpty);
      expect(t.camera.grid, isNotEmpty);
      expect(t.camera.settings, isNotEmpty);
      expect(t.camera.sensors, isNotEmpty);
      expect(t.camera.settingsTitle, isNotEmpty);
      expect(t.camera.resolutions.low, isNotEmpty);
      expect(t.camera.resolutions.medium, isNotEmpty);
      expect(t.camera.resolutions.high, isNotEmpty);
      expect(t.camera.resolutions.veryHigh, isNotEmpty);
      expect(t.camera.resolutions.ultraHigh, isNotEmpty);
      expect(t.camera.resolutions.max, isNotEmpty);
    });

    test('capture namespace keys are non-empty', () {
      expect(t.capture.recording, isNotEmpty);
      expect(t.capture.stop, isNotEmpty);
      expect(t.capture.saving, isNotEmpty);
      expect(t.capture.shareText, isNotEmpty);
    });

    test('export namespace keys are non-empty', () {
      expect(t.export.title, isNotEmpty);
      expect(t.export.success, isNotEmpty);
      expect(t.export.error, isNotEmpty);
      expect(t.export.newCapture, isNotEmpty);
    });

    test('metrics namespace keys are non-empty', () {
      expect(t.metrics.title, isNotEmpty);
      expect(t.metrics.gyro, isNotEmpty);
      expect(t.metrics.accel, isNotEmpty);
      expect(t.metrics.edgeAngle, isNotEmpty);
    });

    test('overlay namespace keys are non-empty', () {
      expect(t.overlay.leftLabel, isNotEmpty);
      expect(t.overlay.rightLabel, isNotEmpty);
      expect(t.overlay.staleAlert, isNotEmpty);
    });
  });
}
