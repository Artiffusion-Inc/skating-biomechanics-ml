import 'dart:async';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:mocktail/mocktail.dart';
import 'package:edgesense_capture/ble/imu_device.dart';

class MockBluetoothDevice extends Mock implements BluetoothDevice {}

void main() {
  group('IMUDevice', () {
    late MockBluetoothDevice mockDevice;
    late StreamController<BluetoothConnectionState> stateController;

    setUp(() {
      mockDevice = MockBluetoothDevice();
      stateController = StreamController<BluetoothConnectionState>.broadcast();
      when(() => mockDevice.isConnected).thenReturn(true);
      when(
        () => mockDevice.connectionState,
      ).thenAnswer((_) => stateController.stream);
      when(() => mockDevice.disconnect()).thenAnswer((_) async {});
    });

    tearDown(() {
      stateController.close();
    });

    test('does not mutate isConnected eagerly on disconnect', () async {
      stateController.add(BluetoothConnectionState.connected);
      final imu = IMUDevice(device: mockDevice, side: 'left');

      expect(imu.isConnected.value, true);

      // disconnect() must not eagerly set false
      await imu.disconnect();
      expect(imu.isConnected.value, true); // still true until stream emits

      // Real disconnection event from stream
      stateController.add(BluetoothConnectionState.disconnected);
      await Future.delayed(const Duration(milliseconds: 10));

      // Now it's false
      expect(imu.isConnected.value, false);

      imu.dispose();
    });

    test('initial isConnected comes from device.isConnected', () {
      when(() => mockDevice.isConnected).thenReturn(false);
      stateController.add(BluetoothConnectionState.disconnected);
      final imu = IMUDevice(device: mockDevice, side: 'left');
      expect(imu.isConnected.value, false);
      imu.dispose();
    });
  });
}
