import 'package:permission_handler/permission_handler.dart';

abstract class PermissionGateway {
  Future<PermissionStatus> requestLocation();
  Future<PermissionStatus> requestBluetoothScan();
  Future<PermissionStatus> requestBluetoothConnect();
  Future<PermissionStatus> requestCamera();
  Future<PermissionStatus> requestMicrophone();

  Future<bool> isLocationGranted();
  Future<bool> isBluetoothScanGranted();
  Future<bool> isBluetoothConnectGranted();
  Future<bool> isCameraGranted();
  Future<bool> isMicrophoneGranted();
}

class PermissionHandlerGateway implements PermissionGateway {
  @override
  Future<PermissionStatus> requestLocation() => Permission.location.request();

  @override
  Future<PermissionStatus> requestBluetoothScan() =>
      Permission.bluetoothScan.request();

  @override
  Future<PermissionStatus> requestBluetoothConnect() =>
      Permission.bluetoothConnect.request();

  @override
  Future<PermissionStatus> requestCamera() => Permission.camera.request();

  @override
  Future<PermissionStatus> requestMicrophone() =>
      Permission.microphone.request();

  @override
  Future<bool> isLocationGranted() => Permission.location.isGranted;

  @override
  Future<bool> isBluetoothScanGranted() => Permission.bluetoothScan.isGranted;

  @override
  Future<bool> isBluetoothConnectGranted() =>
      Permission.bluetoothConnect.isGranted;

  @override
  Future<bool> isCameraGranted() => Permission.camera.isGranted;

  @override
  Future<bool> isMicrophoneGranted() => Permission.microphone.isGranted;
}
