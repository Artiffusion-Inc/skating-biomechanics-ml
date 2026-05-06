import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'wt901_parser.dart';

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
      if (wasConnected != isConnected) onStateChanged?.call();
    });
  }

  Future<void> connect() async {
    await device.connect(autoConnect: false);
  }

  Future<void> disconnect() async {
    await _notifySubscription?.cancel();
    _notifySubscription = null;
    await _connectionSubscription?.cancel();
    _connectionSubscription = null;
    try { await device.disconnect(); } catch (_) {}
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
