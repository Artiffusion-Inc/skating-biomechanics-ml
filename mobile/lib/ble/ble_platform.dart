import 'dart:async';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

abstract class BlePlatform {
  Stream<BluetoothAdapterState> get adapterState;
  BluetoothAdapterState get adapterStateNow;
  Future<void> stopScan();
  Stream<List<ScanResult>> get scanResults;
  Future<void> startScan({
    Duration? timeout,
    AndroidScanMode androidScanMode = AndroidScanMode.lowLatency,
  });
  Future<bool> get locationGranted;
  Future<bool> get bluetoothScanGranted;
  Future<bool> get bluetoothConnectGranted;
}
