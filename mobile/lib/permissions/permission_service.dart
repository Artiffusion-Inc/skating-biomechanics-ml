import 'package:permission_handler/permission_handler.dart';
import 'permission_handler.dart';

class PermissionService {
  final PermissionGateway _gateway;

  PermissionService({PermissionGateway? gateway})
    : _gateway = gateway ?? PermissionHandlerGateway();

  Future<bool> requestAll() async {
    final location = await _gateway.requestLocation();
    final ble = await _gateway.requestBluetoothScan();
    final bleConnect = await _gateway.requestBluetoothConnect();
    final camera = await _gateway.requestCamera();
    final microphone = await _gateway.requestMicrophone();
    return location.isGranted &&
        ble.isGranted &&
        bleConnect.isGranted &&
        camera.isGranted &&
        microphone.isGranted;
  }

  static Future<bool> allGranted() async {
    return await Permission.location.isGranted &&
        await Permission.bluetoothScan.isGranted &&
        await Permission.bluetoothConnect.isGranted &&
        await Permission.camera.isGranted &&
        await Permission.microphone.isGranted;
  }
}
