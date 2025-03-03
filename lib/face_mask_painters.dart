import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(this.imageSize, this.results);

  final Size imageSize;
  late double scaleX;
  late double scaleY;
  late dynamic results;

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.greenAccent;

    for (String label in results.keys) {
      for (Face face in results[label]) {
        // Calculate scaling factors
        scaleX = size.width / imageSize.width;
        scaleY = size.height / imageSize.height;

        // Draw the bounding box
        canvas.drawRRect(
          _scaleRect(
            rect: face.boundingBox,
            imageSize: imageSize,
            widgetSize: size,
            scaleX: scaleX,
            scaleY: scaleY,
          ),
          paint,
        );

        // Calculate the center of the bounding box
        final centerX = size.width -
            (face.boundingBox.left + face.boundingBox.width / 2) * scaleX;
        final centerY =
            (face.boundingBox.top + face.boundingBox.height / 2) * scaleY;

        // Draw the label at the center of the bounding box
        TextSpan span = TextSpan(
          style: TextStyle(color: Colors.orange[300], fontSize: 15),
          text: label,
        );
        TextPainter textPainter = TextPainter(
          text: span,
          textAlign: TextAlign.center, // Center the text
          textDirection: TextDirection.ltr,
        );
        textPainter.layout();

        // Position the text at the center of the bounding box
        textPainter.paint(
          canvas,
          Offset(
            centerX - textPainter.width / 2, // Center horizontally
            centerY - textPainter.height / 2, // Center vertically
          ),
        );
      }
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.results != results;
  }
}

RRect _scaleRect({
  required Rect? rect,
  required Size? imageSize,
  required Size? widgetSize,
  double? scaleX,
  double? scaleY,
}) {
  return RRect.fromLTRBR(
    widgetSize!.width - rect!.left.toDouble() * scaleX!,
    rect.top.toDouble() * scaleY!,
    widgetSize.width - rect.right.toDouble() * scaleX,
    rect.bottom.toDouble() * scaleY,
    Radius.circular(10),
  );
}