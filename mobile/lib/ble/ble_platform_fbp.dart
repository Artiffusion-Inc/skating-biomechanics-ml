import 'dart:async';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import 'ble_platform.dart';

class BlePlatformFbp implements BlePlatform {
  @override
  Stream<BluetoothAdapterState> get adapterState =>
      FlutterBluePlus.adapterState;

  @override
  BluetoothAdapterState get adapterStateNow => FlutterBluePlus.adapterStateNow;

  @override
  Future<void> stopScan() => FlutterBluePlus.stopScan();

  @override
  Stream<List<ScanResult>> get scanResults => FlutterBluePlus.scanResults;

  @override
  Future<void> startScan({
    Duration? timeout,
    AndroidScanMode androidScanMode = AndroidScanMode.lowLatency,
  }) => FlutterBluePlus.startScan(
    timeout: timeout,
    androidScanMode: androidScanMode,
  );

  @override
  Future<bool> get locationGranted => Permission.location.isGranted;

  @override
  Future<bool> get bluetoothScanGranted => Permission.bluetoothScan.isGranted;

  @override
  Future<bool> get bluetoothConnectGranted =>
      Permission.bluetoothConnect.isGranted;
}
