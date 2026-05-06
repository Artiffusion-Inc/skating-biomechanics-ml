import 'dart:async';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:edgesense_capture/ble/ble_manager.dart';

void main() {
  group('BleManager', () {
    test('initializes with empty scan results', () {
      final manager = BleManager();
      expect(manager.scanResults, isEmpty);
      expect(manager.leftDevice, isNull);
      expect(manager.rightDevice, isNull);
    });

    test('assigns left device', () {
      final manager = BleManager();
      final device = BluetoothDevice.fromId('AA:BB:CC:DD:EE:01');
      manager.assignDevice('left', device);
      expect(manager.leftDevice, isNotNull);
      expect(manager.leftDevice!.side, equals('left'));
    });

    test('assigns right device', () {
      final manager = BleManager();
      final device = BluetoothDevice.fromId('AA:BB:CC:DD:EE:02');
      manager.assignDevice('right', device);
      expect(manager.rightDevice, isNotNull);
      expect(manager.rightDevice!.side, equals('right'));
    });

    test('overwrites left device on reassign', () {
      final manager = BleManager();
      final device1 = BluetoothDevice.fromId('AA:BB:CC:DD:EE:01');
      final device2 = BluetoothDevice.fromId('AA:BB:CC:DD:EE:03');
      manager.assignDevice('left', device1);
      manager.assignDevice('left', device2);
      expect(manager.leftDevice!.device.remoteId.str, equals('AA:BB:CC:DD:EE:03'));
    });
  });
}
