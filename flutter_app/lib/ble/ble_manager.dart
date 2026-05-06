import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'wt901_parser.dart';
import 'package:async/async.dart';

class IMUDevice {
  final BluetoothDevice device;
  final String side; // 'left' or 'right'
  StreamSubscription? _notifySubscription;

  IMUDevice({required this.device, required this.side});

  Future<void> connect() async {
    await device.connect(autoConnect: false);
  }

  Future<void> disconnect() async {
    await _notifySubscription?.cancel();
    await device.disconnect();
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
  IMUDevice? leftDevice;
  IMUDevice? rightDevice;

  StreamSubscription? _scanSubscription;

  Future<void> startScan() async {
    await FlutterBluePlus.startScan(timeout: const Duration(seconds: 4));
    _scanSubscription = FlutterBluePlus.scanResults.listen((results) {
      scanResults = results;
    });
  }

  Future<void> stopScan() async {
    await _scanSubscription?.cancel();
    await FlutterBluePlus.stopScan();
  }

  void assignDevice(String side, BluetoothDevice device) {
    final imuDevice = IMUDevice(device: device, side: side);
    if (side == 'left') {
      leftDevice = imuDevice;
    } else {
      rightDevice = imuDevice;
    }
    notifyListeners();
  }

  void unassignDevice(String side) {
    if (side == 'left') {
      leftDevice = null;
    } else {
      rightDevice = null;
    }
    notifyListeners();
  }

  Future<void> connectAll() async {
    await Future.wait([
      if (leftDevice != null) leftDevice!.connect(),
      if (rightDevice != null) rightDevice!.connect(),
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
}
