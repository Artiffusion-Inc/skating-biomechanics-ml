import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../ble/ble_manager.dart';
import '../ble/wt901_parser.dart';
import 'calibration_service.dart';

class CalibrationScreen extends StatefulWidget {
  final VoidCallback onComplete;
  const CalibrationScreen({super.key, required this.onComplete});

  @override
  State<CalibrationScreen> createState() => _CalibrationScreenState();
}

class _CalibrationScreenState extends State<CalibrationScreen> {
  static const _calibDuration = Duration(seconds: 3);
  bool _running = false;
  bool _done = false;
  double _progress = 0;
  String? _error;
  StreamSubscription? _sub;
  final List<List<double>> _leftQuats = [];
  final List<List<double>> _rightQuats = [];

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  Future<void> _startCalibration() async {
    setState(() {
      _running = true;
      _done = false;
      _error = null;
      _progress = 0;
      _leftQuats.clear();
      _rightQuats.clear();
    });

    final ble = context.read<BleManager>();
    try {
      await ble.connectAll();
    } catch (e) {
      setState(() {
        _running = false;
        _error = 'Ошибка подключения: $e';
      });
      return;
    }

    final stream = ble.startStreams();
    final sw = Stopwatch()..start();

    _sub = stream.listen((pair) {
      final left = pair.$1;
      final right = pair.$2;

      if (left != null && left.quatW != null) {
        _leftQuats.add([left.quatW!, left.quatX!, left.quatY!, left.quatZ!]);
      }
      if (right != null && right.quatW != null) {
        _rightQuats.add([right.quatW!, right.quatX!, right.quatY!, right.quatZ!]);
      }

      final elapsed = sw.elapsed;
      if (mounted) {
        setState(() {
          _progress = (elapsed.inMilliseconds / _calibDuration.inMilliseconds).clamp(0.0, 1.0);
        });
      }

      if (elapsed >= _calibDuration) {
        _sub?.cancel();
        _finish();
      }
    });
  }

  void _finish() {
    final calibration = context.read<CalibrationService>();

    if (_leftQuats.isNotEmpty) {
      calibration.leftRef = calibration.calibrate(_leftQuats);
    }
    if (_rightQuats.isNotEmpty) {
      calibration.rightRef = calibration.calibrate(_rightQuats);
    }

    if (mounted) {
      setState(() {
        _running = false;
        _done = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Калибровка')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.straighten, size: 64, color: Colors.white54),
              const SizedBox(height: 24),
              Text(
                _done
                    ? 'Калибровка завершена'
                    : _running
                        ? 'Стойте неподвижно...'
                        : 'Удерживайте датчики неподвижно\nдля калибровки нулевого угла',
                style: Theme.of(context).textTheme.titleMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              if (_running) ...[
                SizedBox(
                  width: 200,
                  child: LinearProgressIndicator(value: _progress),
                ),
                const SizedBox(height: 8),
                Text(
                  '${(_progress * _calibDuration.inSeconds).toStringAsFixed(1)} / ${_calibDuration.inSeconds} сек',
                  style: const TextStyle(color: Colors.white70),
                ),
              ],
              if (_error != null) ...[
                const SizedBox(height: 16),
                Text(_error!, style: const TextStyle(color: Colors.red)),
              ],
              if (_done) ...[
                const SizedBox(height: 16),
                _CalibResult(label: 'Левый', ref: context.watch<CalibrationService>().leftRef),
                const SizedBox(height: 8),
                _CalibResult(label: 'Правый', ref: context.watch<CalibrationService>().rightRef),
                const SizedBox(height: 24),
                FilledButton.icon(
                  onPressed: widget.onComplete,
                  icon: const Icon(Icons.arrow_forward),
                  label: const Text('Начать запись'),
                ),
              ],
              if (!_running && !_done) ...[
                const SizedBox(height: 24),
                FilledButton.icon(
                  onPressed: _startCalibration,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Калибровать'),
                ),
                const SizedBox(height: 12),
                TextButton(
                  onPressed: widget.onComplete,
                  child: const Text('Пропустить'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _CalibResult extends StatelessWidget {
  final String label;
  final List<double>? ref;
  const _CalibResult({required this.label, required this.ref});

  @override
  Widget build(BuildContext context) {
    if (ref == null) return Text('$label: нет данных', style: const TextStyle(color: Colors.white54));
    return Text(
      '$label: q=[${ref![0].toStringAsFixed(3)}, ${ref![1].toStringAsFixed(3)}, ${ref![2].toStringAsFixed(3)}, ${ref![3].toStringAsFixed(3)}]',
      style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
    );
  }
}