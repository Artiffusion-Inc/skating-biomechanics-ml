import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import 'wt901_parser.dart';
import 'package:async/async.dart';

class IMUDevice {
  final BluetoothDevice device;
  final String side;
  bool isConnected = false;
  final VoidCallback? onStateChanged;
  StreamSubscription? _notifySubscription;
  StreamSubscription? _connectionSubscription;

  IMUDevice({required this.device, required this.side, this.onStateChanged}) {
    isConnected = device.isConnected;
    _connectionSubscription = device.connectionState.listen((state) {
      final wasConnected = isConnected;
      isConnected = state == BluetoothConnectionState.connected;
      print('[BLE] ${device.platformName} state: $state (isConnected=$isConnected)');
      if (wasConnected != isConnected) onStateChanged?.call();
    });
  }

  Future<void> connect() async {
    print('[BLE] IMUDevice.connect() called for ${device.platformName}');
    await device.connect(autoConnect: false);
    print('[BLE] IMUDevice.connect() completed for ${device.platformName}');
  }

  Future<void> disconnect() async {
    await _notifySubscription?.cancel();
    _notifySubscription = null;
    await _connectionSubscription?.cancel();
    _connectionSubscription = null;
    try {
      await device.disconnect();
    } catch (_) {}
    isConnected = false;
  }

  Stream<WT901Packet> startNotifications() async* {
    final services = await device.discoverServices();
    final targetService = services.firstWhere(
      (s) => s.uuid.toString().toLowerCase().contains('ffe0'),
      orElse: () => services.first,
    );
    final characteristic = targetService.characteristics.firstWhere(
      (c) => c.properties.notify,
    );
    await characteristic.setNotifyValue(true);
    await for (final event in characteristic.lastValueStream) {
      final packet = WT901Parser.parse(event);
      if (packet != null) yield packet;
    }
  }
}

class BleManager extends ChangeNotifier {
  List<ScanResult> scanResults = [];
  List<ScanResult> get namedScanResults =>
      scanResults.where((r) => r.device.platformName.isNotEmpty).toList();
  IMUDevice? leftDevice;
  IMUDevice? rightDevice;
  bool isScanning = false;
  String? scanError;
  bool locationPermissionGranted = false;

  StreamSubscription? _scanSubscription;
  StreamSubscription? _adapterSubscription;

  BleManager() {
    _adapterSubscription = FlutterBluePlus.adapterState.listen((state) {
      if (state == BluetoothAdapterState.off) {
        stopScan();
      }
      notifyListeners();
    });
  }

  bool get isBluetoothOn =>
      FlutterBluePlus.adapterStateNow == BluetoothAdapterState.on;

  bool get canProceed =>
      (leftDevice?.isConnected ?? false) || (rightDevice?.isConnected ?? false);

  Future<bool> checkLocationPermission() async {
    locationPermissionGranted = await Permission.location.isGranted;
    notifyListeners();
    return locationPermissionGranted;
  }

  Future<bool> checkBluetoothPermissions() async =>
      await Permission.bluetoothScan.isGranted &&
      await Permission.bluetoothConnect.isGranted;

  DateTime? _lastScanStop;

  Future<void> startScan() async {
    if (isScanning) return;
    if (!isBluetoothOn) {
      scanError = 'Bluetooth выключен';
      notifyListeners();
      return;
    }
    if (!await checkLocationPermission()) {
      scanError = 'Требуется разрешение на местоположение для BLE сканирования';
      notifyListeners();
      return;
    }

    scanError = null;

    // Throttle: Android blocks scan if restarted within ~5s
    if (_lastScanStop != null) {
      final elapsed = DateTime.now().difference(_lastScanStop!);
      if (elapsed.inSeconds < 6) {
        await Future.delayed(Duration(seconds: 6 - elapsed.inSeconds));
      }
    }

    // Ensure previous scan fully stopped
    await FlutterBluePlus.stopScan();
    await _scanSubscription?.cancel();
    _scanSubscription = null;
    scanResults = [];
    notifyListeners();

    isScanning = true;
    notifyListeners();

    _scanSubscription = FlutterBluePlus.scanResults.listen((results) {
      scanResults = results;
      notifyListeners();
    });

    try {
      await FlutterBluePlus.startScan(
        timeout: const Duration(seconds: 10),
        androidScanMode: AndroidScanMode.lowLatency,
      );
    } catch (e) {
      scanError = e.toString();
    }

    isScanning = false;
    _lastScanStop = DateTime.now();
    notifyListeners();
  }

  Future<void> stopScan() async {
    await _scanSubscription?.cancel();
    _scanSubscription = null;
    await FlutterBluePlus.stopScan();
    isScanning = false;
    _lastScanStop = DateTime.now();
    notifyListeners();
  }

  void assignDevice(String side, BluetoothDevice device) {
    final imu = IMUDevice(device: device, side: side, onStateChanged: notifyListeners);
    if (side == 'left') {
      leftDevice?.disconnect();
      leftDevice = imu;
    } else {
      rightDevice?.disconnect();
      rightDevice = imu;
    }
    // Auto-connect after assigning
    _connectDevice(imu);
    notifyListeners();
  }

  Future<void> _connectDevice(IMUDevice imu) async {
    print('[BLE] Connecting to ${imu.device.platformName} (${imu.device.id.id})...');
    try {
      await imu.connect();
      print('[BLE] Connected to ${imu.device.platformName}');
    } catch (e) {
      print('[BLE] Connection failed: $e');
    }
  }

  void unassignDevice(String side) {
    if (side == 'left') {
      leftDevice?.disconnect();
      leftDevice = null;
    } else {
      rightDevice?.disconnect();
      rightDevice = null;
    }
    notifyListeners();
  }

  Future<void> connectAll() async {
    await Future.wait([
      if (leftDevice != null && !leftDevice!.isConnected) leftDevice!.connect(),
      if (rightDevice != null && !rightDevice!.isConnected) rightDevice!.connect(),
    ]);
  }

  Future<void> disconnectAll() async {
    await Future.wait([
      if (leftDevice != null) leftDevice!.disconnect(),
      if (rightDevice != null) rightDevice!.disconnect(),
    ]);
  }

  Stream<(WT901Packet?, WT901Packet?)> startStreams() {
    final leftStream = leftDevice?.startNotifications() ?? Stream<WT901Packet>.empty();
    final rightStream = rightDevice?.startNotifications() ?? Stream<WT901Packet>.empty();
    return StreamZip([leftStream, rightStream]).map((pair) => (pair[0], pair[1]));
  }

  @override
  void dispose() {
    _scanSubscription?.cancel();
    _adapterSubscription?.cancel();
    super.dispose();
  }
}