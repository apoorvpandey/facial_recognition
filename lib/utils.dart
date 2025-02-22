import 'dart:async';
import 'dart:io';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class ScannerUtils {
  ScannerUtils._();

  static Future<CameraDescription> getCamera(CameraLensDirection dir) async {
    return availableCameras().then(
          (List<CameraDescription> cameras) => cameras.firstWhere(
            (CameraDescription camera) => camera.lensDirection == dir,
      ),
    );
  }

  static Future<dynamic> detect({
    required CameraImage image,
    required Future<dynamic> Function(InputImage inputImage) detectInImage,
    required InputImageRotation imageRotation,
    required CameraController cameraController,
  }) async {
    final inputImage =
    _convertCameraImageToInputImage(image, imageRotation, cameraController);
    if (inputImage == null) {
      throw Exception("Failed to convert CameraImage to InputImage");
    }
    return detectInImage(inputImage);
  }

  static InputImage? _convertCameraImageToInputImage(
      CameraImage cameraImage,
      InputImageRotation imageRotation,
      CameraController cameraController,
      ) {
    try {
      // Determine the image format based on the platform
      final InputImageFormat format = Platform.isAndroid
          ? InputImageFormat.nv21
          : InputImageFormat.bgra8888;

      // Handle YUV format (Android)
      if (Platform.isAndroid &&
          cameraImage.format.group == ImageFormatGroup.yuv420) {
        final WriteBuffer allBytes = WriteBuffer();
        for (final Plane plane in cameraImage.planes) {
          allBytes.putUint8List(plane.bytes);
        }
        final bytes = allBytes.done().buffer.asUint8List();

        return InputImage.fromBytes(
          bytes: bytes,
          metadata: InputImageMetadata(
            size: Size(
                cameraImage.width.toDouble(), cameraImage.height.toDouble()),
            rotation: imageRotation,
            format: format,
            bytesPerRow: cameraImage.planes.first.bytesPerRow,
          ),
        );
      }

      // Handle single plane format (iOS)
      if (cameraImage.planes.length != 1) {
        if (kDebugMode) {
          print("❌ Unsupported number of planes: ${cameraImage.planes.length}");
        }
        return null;
      }

      final Plane plane = cameraImage.planes.first;
      return InputImage.fromBytes(
        bytes: plane.bytes,
        metadata: InputImageMetadata(
          size:
          Size(cameraImage.width.toDouble(), cameraImage.height.toDouble()),
          rotation: imageRotation,
          format: format,
          bytesPerRow: plane.bytesPerRow,
        ),
      );
    } catch (e) {
      if (kDebugMode) {
        print("❌ Error converting CameraImage to InputImage: $e");
      }
      return null;
    }
  }
}
