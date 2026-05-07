import 'package:permission_handler/permission_handler.dart';

class PermissionService {
  Future<bool> requestAll() async {
    final location = await Permission.location.request();
    final ble = await Permission.bluetoothScan.request();
    final bleConnect = await Permission.bluetoothConnect.request();
    final camera = await Permission.camera.request();
    final microphone = await Permission.microphone.request();
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
