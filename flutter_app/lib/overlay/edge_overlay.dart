import 'package:flutter/material.dart';

class EdgeOverlay extends StatelessWidget {
  final double leftAngle;
  final double rightAngle;

  const EdgeOverlay({
    super.key,
    required this.leftAngle,
    required this.rightAngle,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 40,
      left: 20,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('L: ${leftAngle.toStringAsFixed(1)}°',
                style: const TextStyle(color: Colors.white, fontSize: 18)),
            Text('R: ${rightAngle.toStringAsFixed(1)}°',
                style: const TextStyle(color: Colors.white, fontSize: 18)),
          ],
        ),
      ),
    );
  }
}
