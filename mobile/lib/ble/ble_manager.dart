import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import 'wt901_parser.dart';
import 'imu_device.dart';

enum BleScanError {
  bluetoothOff,
  locationRequired,
  unknown,
}

class BleManager extends ChangeNotifier {
  List<ScanResult> scanResults = [];
  List<ScanResult> get namedScanResults =>
      scanResults.where((r) => r.device.platformName.isNotEmpty).toList();
  IMUDevice? leftDevice;
  IMUDevice? rightDevice;
  bool isScanning = false;
  BleScanError? scanError;
  String? scanErrorMessage;
  bool locationPermissionGranted = false;

  final Map<String, double> batteryLevels = {};

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
      (leftDevice?.isConnected.value ?? false) || (rightDevice?.isConnected.value ?? false);

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
      scanError = BleScanError.bluetoothOff;
      scanErrorMessage = null;
      notifyListeners();
      return;
    }
    if (!await checkLocationPermission()) {
      scanError = BleScanError.locationRequired;
      scanErrorMessage = null;
      notifyListeners();
      return;
    }

    scanError = null;
    scanErrorMessage = null;

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
      scanError = BleScanError.unknown;
      scanErrorMessage = e.toString();
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
    final imu = IMUDevice(
      device: device,
      side: side,
      onBattery: (v) {
        batteryLevels[device.remoteId.str] = v;
        notifyListeners();
      },
    );
    // Subscribe to connection state changes
    imu.isConnected.addListener(notifyListeners);

    if (side == 'left') {
      leftDevice?.dispose();
      leftDevice = imu;
    } else {
      rightDevice?.dispose();
      rightDevice = imu;
    }
    // Auto-connect after assigning
    _connectDevice(imu);
    notifyListeners();
  }

  Future<void> _connectDevice(IMUDevice imu) async {
    try {
      await imu.connect();
    } catch (e) {
      // Connection will be retried or user can tap to retry
    }
  }

  void unassignDevice(String side) {
    if (side == 'left') {
      leftDevice?.dispose();
      leftDevice = null;
    } else {
      rightDevice?.dispose();
      rightDevice = null;
    }
    notifyListeners();
  }

  Future<void> connectAll() async {
    await Future.wait([
      if (leftDevice != null && !leftDevice!.isConnected.value) leftDevice!.connect(),
      if (rightDevice != null && !rightDevice!.isConnected.value) rightDevice!.connect(),
    ]);
  }

  Future<void> disconnectAll() async {
    await Future.wait([
      if (leftDevice != null) leftDevice!.disconnect(),
      if (rightDevice != null) rightDevice!.disconnect(),
    ]);
  }

  Stream<(WT901Packet?, WT901Packet?)> startStreams() {
    final controller = StreamController<(WT901Packet?, WT901Packet?)>.broadcast();
    WT901Packet? lastLeft, lastRight;
    StreamSubscription? leftSub, rightSub;
    Timer? timer;

    void emit() {
      if (!controller.isClosed) controller.add((lastLeft, lastRight));
    }

    if (leftDevice != null && leftDevice!.isConnected.value) {
      leftSub = leftDevice!.startNotifications().listen((p) {
        lastLeft = p;
        emit();
      });
    }
    if (rightDevice != null && rightDevice!.isConnected.value) {
      rightSub = rightDevice!.startNotifications().listen((p) {
        lastRight = p;
        emit();
      });
    }

    timer = Timer.periodic(const Duration(milliseconds: 100), (_) => emit());

    controller.onCancel = () {
      timer?.cancel();
      leftSub?.cancel();
      rightSub?.cancel();
    };

    return controller.stream;
  }

  @override
  void dispose() {
    _scanSubscription?.cancel();
    _adapterSubscription?.cancel();
    leftDevice?.dispose();
    rightDevice?.dispose();
    super.dispose();
  }
}
