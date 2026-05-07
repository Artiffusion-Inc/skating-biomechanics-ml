import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:edgesense_capture/overlay/edge_overlay.dart';

void main() {
  testWidgets('displays left and right edge angles', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: Stack(
            children: [
              EdgeOverlay(leftAngle: 15.0, rightAngle: -10.0),
            ],
          ),
        ),
      ),
    );
    expect(find.text('L: 15.0°'), findsOneWidget);
    expect(find.text('R: -10.0°'), findsOneWidget);
  });
}
