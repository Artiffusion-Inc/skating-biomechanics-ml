import 'package:flutter/material.dart';

class EdgeOverlay extends StatelessWidget {
  final double leftAngle;
  final double rightAngle;
  final bool leftActive;
  final bool rightActive;

  const EdgeOverlay({
    super.key,
    required this.leftAngle,
    required this.rightAngle,
    this.leftActive = false,
    this.rightActive = false,
  });

  @override
  Widget build(BuildContext context) {
    final leftStale = !leftActive;
    final rightStale = !rightActive;

    return Positioned(
      top: 40,
      left: 20,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(8),
          border: (leftStale || rightStale)
              ? Border.all(color: Colors.red.shade700, width: 2)
              : null,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                if (leftStale) ...[
                  Icon(Icons.warning_amber, color: Colors.red.shade400, size: 16),
                  const SizedBox(width: 4),
                ],
                Text(
                  'L: ${leftAngle.toStringAsFixed(1)}°',
                  style: TextStyle(
                    color: leftStale ? Colors.red.shade300 : Colors.white,
                    fontSize: 18,
                    fontWeight: leftStale ? FontWeight.bold : null,
                  ),
                ),
              ],
            ),
            Row(
              children: [
                if (rightStale) ...[
                  Icon(Icons.warning_amber, color: Colors.red.shade400, size: 16),
                  const SizedBox(width: 4),
                ],
                Text(
                  'R: ${rightAngle.toStringAsFixed(1)}°',
                  style: TextStyle(
                    color: rightStale ? Colors.red.shade300 : Colors.white,
                    fontSize: 18,
                    fontWeight: rightStale ? FontWeight.bold : null,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
