import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'wt901_parser.dart';

class IMUDevice {
  final BluetoothDevice device;
  final String side;
  final void Function(double voltage)? onBattery;

  final ValueNotifier<bool> isConnected = ValueNotifier<bool>(false);

  StreamSubscription? _notifySubscription;
  StreamSubscription? _connectionSubscription;

  IMUDevice({required this.device, required this.side, this.onBattery}) {
    isConnected.value = device.isConnected;
    _connectionSubscription = device.connectionState.listen((state) {
      final connected = state == BluetoothConnectionState.connected;
      if (isConnected.value != connected) {
        isConnected.value = connected;
      }
    });
  }

  Future<void> connect() async {
    await device.connect(autoConnect: false);
  }

  Future<void> disconnect() async {
    await _notifySubscription?.cancel();
    _notifySubscription = null;
    // Do NOT mutate isConnected.value here. Let device.connectionState stream do it.
    try {
      await device.disconnect();
    } catch (_) {}
    // Do NOT cancel _connectionSubscription here — the platform stack may still
    // emit a disconnected event we want to observe.
  }

  void dispose() {
    _connectionSubscription?.cancel();
    isConnected.dispose();
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
      if (packet != null) {
        if (packet.type == WT901PacketType.battery && packet.battery != null) {
          onBattery?.call(packet.battery!);
        }
        yield packet;
      }
    }
  }
}
