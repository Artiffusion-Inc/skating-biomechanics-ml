import 'dart:async';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:edgesense_capture/ble/ble_manager.dart';
import 'package:edgesense_capture/ble/ble_platform.dart';

class MockBlePlatform extends Mock implements BlePlatform {}

class MockBluetoothDevice extends Mock implements BluetoothDevice {}

class FakeAndroidScanMode extends Fake implements AndroidScanMode {}

void main() {
  setUpAll(() {
    registerFallbackValue(FakeAndroidScanMode());
  });

  group('BleManager', () {
    late MockBlePlatform platform;
    late StreamController<BluetoothAdapterState> adapterController;
    late StreamController<List<ScanResult>> scanResultsController;
    late BleManager manager;

    setUp(() {
      platform = MockBlePlatform();
      adapterController = StreamController<BluetoothAdapterState>.broadcast();
      scanResultsController = StreamController<List<ScanResult>>.broadcast();

      when(
        () => platform.adapterState,
      ).thenAnswer((_) => adapterController.stream);
      when(() => platform.adapterStateNow).thenReturn(BluetoothAdapterState.on);
      when(() => platform.stopScan()).thenAnswer((_) async {});
      when(
        () => platform.scanResults,
      ).thenAnswer((_) => scanResultsController.stream);
      when(
        () => platform.startScan(
          timeout: any(named: 'timeout'),
          androidScanMode: any(named: 'androidScanMode'),
        ),
      ).thenAnswer((_) async {});
      when(() => platform.locationGranted).thenAnswer((_) async => true);
      when(() => platform.bluetoothScanGranted).thenAnswer((_) async => true);
      when(
        () => platform.bluetoothConnectGranted,
      ).thenAnswer((_) async => true);

      manager = BleManager(platform: platform);
    });

    tearDown(() {
      adapterController.close();
      scanResultsController.close();
      manager.dispose();
    });

    test('initial state', () {
      expect(manager.isScanning, false);
      expect(manager.scanResults, isEmpty);
      expect(manager.leftDevice, isNull);
      expect(manager.rightDevice, isNull);
      expect(manager.scanError, isNull);
    });

    test('checkLocationPermission updates state and notifies', () async {
      when(() => platform.locationGranted).thenAnswer((_) async => true);
      var notified = false;
      manager.addListener(() => notified = true);

      final result = await manager.checkLocationPermission();

      expect(result, true);
      expect(manager.locationPermissionGranted, true);
      expect(notified, true);
    });

    test('startScan sets bluetoothOff error when adapter is off', () async {
      when(
        () => platform.adapterStateNow,
      ).thenReturn(BluetoothAdapterState.off);
      await manager.startScan();

      expect(manager.scanError, BleScanError.bluetoothOff);
      expect(manager.isScanning, false);
    });

    test(
      'startScan sets locationRequired error when permission denied',
      () async {
        when(() => platform.locationGranted).thenAnswer((_) async => false);
        await manager.startScan();

        expect(manager.scanError, BleScanError.locationRequired);
        expect(manager.isScanning, false);
      },
    );

    test('startScan success lifecycle', () async {
      final scanResult = ScanResult(
        device: MockBluetoothDevice(),
        advertisementData: AdvertisementData(
          advName: 'Test',
          txPowerLevel: null,
          appearance: null,
          connectable: true,
          manufacturerData: {},
          serviceData: {},
          serviceUuids: [],
        ),
        rssi: -50,
        timeStamp: DateTime.now(),
      );

      manager.addListener(() {});

      await manager.startScan();
      expect(manager.isScanning, false);

      scanResultsController.add([scanResult]);
      await Future.delayed(Duration.zero);

      expect(manager.scanResults.length, 1);
    });

    test('stopScan cancels subscription and stops platform scan', () async {
      await manager.stopScan();

      verify(() => platform.stopScan()).called(1);
      expect(manager.isScanning, false);
    });

    test('adapter off stops active scan', () async {
      adapterController.add(BluetoothAdapterState.off);
      await Future.delayed(Duration.zero);

      verify(() => platform.stopScan()).called(greaterThanOrEqualTo(1));
      expect(manager.isScanning, false);
    });

    test('assignDevice left and battery callback', () async {
      final device = MockBluetoothDevice();
      when(
        () => device.remoteId,
      ).thenReturn(DeviceIdentifier('aa:bb:cc:dd:ee:ff'));
      when(() => device.isConnected).thenReturn(true);
      when(
        () => device.connectionState,
      ).thenAnswer((_) => Stream.value(BluetoothConnectionState.connected));
      when(() => device.disconnect()).thenAnswer((_) async {});
      when(() => device.connect(autoConnect: false)).thenAnswer((_) async {});

      var notified = false;
      manager.addListener(() => notified = true);

      manager.assignDevice('left', device);

      expect(manager.leftDevice, isNotNull);
      expect(manager.leftDevice!.side, 'left');
      expect(notified, true);

      manager.leftDevice!.onBattery?.call(3.8);
      expect(manager.batteryLevels['aa:bb:cc:dd:ee:ff'], 3.8);
    });

    test('assignDevice right', () async {
      final device = MockBluetoothDevice();
      when(
        () => device.remoteId,
      ).thenReturn(DeviceIdentifier('11:22:33:44:55:66'));
      when(() => device.isConnected).thenReturn(false);
      when(
        () => device.connectionState,
      ).thenAnswer((_) => Stream.value(BluetoothConnectionState.disconnected));
      when(() => device.disconnect()).thenAnswer((_) async {});
      when(() => device.connect(autoConnect: false)).thenAnswer((_) async {});

      manager.assignDevice('right', device);

      expect(manager.rightDevice, isNotNull);
      expect(manager.rightDevice!.side, 'right');
    });

    test('unassignDevice disposes and clears', () async {
      final device = MockBluetoothDevice();
      when(
        () => device.remoteId,
      ).thenReturn(DeviceIdentifier('aa:bb:cc:dd:ee:ff'));
      when(() => device.isConnected).thenReturn(true);
      when(
        () => device.connectionState,
      ).thenAnswer((_) => Stream.value(BluetoothConnectionState.connected));
      when(() => device.disconnect()).thenAnswer((_) async {});
      when(() => device.connect(autoConnect: false)).thenAnswer((_) async {});

      manager.assignDevice('left', device);
      expect(manager.leftDevice, isNotNull);

      manager.unassignDevice('left');
      expect(manager.leftDevice, isNull);
    });

    test('canProceed true when left connected', () async {
      final device = MockBluetoothDevice();
      when(
        () => device.remoteId,
      ).thenReturn(DeviceIdentifier('aa:bb:cc:dd:ee:ff'));
      when(() => device.isConnected).thenReturn(true);
      when(
        () => device.connectionState,
      ).thenAnswer((_) => Stream.value(BluetoothConnectionState.connected));
      when(() => device.disconnect()).thenAnswer((_) async {});
      when(() => device.connect(autoConnect: false)).thenAnswer((_) async {});

      expect(manager.canProceed, false);
      manager.assignDevice('left', device);
      expect(manager.canProceed, true);
    });

    test('connectAll connects disconnected devices', () async {
      final left = MockBluetoothDevice();
      when(
        () => left.remoteId,
      ).thenReturn(DeviceIdentifier('aa:bb:cc:dd:ee:ff'));
      when(() => left.isConnected).thenReturn(false);
      when(
        () => left.connectionState,
      ).thenAnswer((_) => Stream.value(BluetoothConnectionState.disconnected));
      when(() => left.connect(autoConnect: false)).thenAnswer((_) async {});
      when(() => left.disconnect()).thenAnswer((_) async {});

      manager.assignDevice('left', left);
      await manager.connectAll();

      // assignDevice auto-connects + connectAll explicitly connects
      verify(() => left.connect(autoConnect: false)).called(2);
    });

    test('disconnectAll disconnects all devices', () async {
      final left = MockBluetoothDevice();
      when(
        () => left.remoteId,
      ).thenReturn(DeviceIdentifier('aa:bb:cc:dd:ee:ff'));
      when(() => left.isConnected).thenReturn(true);
      when(
        () => left.connectionState,
      ).thenAnswer((_) => Stream.value(BluetoothConnectionState.connected));
      when(() => left.disconnect()).thenAnswer((_) async {});
      when(() => left.connect(autoConnect: false)).thenAnswer((_) async {});

      manager.assignDevice('left', left);
      await manager.disconnectAll();

      verify(() => left.disconnect()).called(1);
    });
  });
}
