import 'package:permission_handler/permission_handler.dart';

class PermissionService {
  Future<bool> requestAll() async {
    final ble = await Permission.bluetoothScan.request();
    final bleConnect = await Permission.bluetoothConnect.request();
    final camera = await Permission.camera.request();
    final microphone = await Permission.microphone.request();
    return ble.isGranted &&
        bleConnect.isGranted &&
        camera.isGranted &&
        microphone.isGranted;
  }
}
