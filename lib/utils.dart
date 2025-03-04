import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as img_lib;

typedef HandleDetection = Future<dynamic> Function(InputImage image);

Future<CameraDescription> getCamera(CameraLensDirection dir) async {
  return await availableCameras().then(
    (List<CameraDescription> cameras) => cameras.firstWhere(
      (CameraDescription camera) => camera.lensDirection == dir,
    ),
  );
}

InputImageMetadata buildMetaData(
  CameraImage image,
  InputImageRotation rotation,
) {
  return InputImageMetadata(
    format: InputImageFormat.yuv420,
    size: Size(image.width.toDouble(), image.height.toDouble()),
    rotation: rotation,
    bytesPerRow: image.planes.first.bytesPerRow,
  );
}

Future<dynamic> detect(
  CameraImage image,
  HandleDetection handleDetection,
  InputImageRotation rotation,
  CameraController? camera,
) async {
  // Create InputImage using a helper function
  final inputImage = _convertCameraImageToInputImage(image, rotation, camera!);

  // Process the image using the provided handleDetection function
  return handleDetection(inputImage!);
}

// Helper function to create InputImage
InputImage? _convertCameraImageToInputImage(
  CameraImage cameraImage,
  InputImageRotation imageRotation,
  CameraController cameraController,
) {
  try {
// Determine the image format based on the platform
    final InputImageFormat format =
        Platform.isAndroid ? InputImageFormat.nv21 : InputImageFormat.bgra8888;

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
          size:
              Size(cameraImage.width.toDouble(), cameraImage.height.toDouble()),
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
        size: Size(cameraImage.width.toDouble(), cameraImage.height.toDouble()),
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

InputImageRotation rotationIntToImageRotation(int rotation) {
  switch (rotation) {
    case 0:
      return InputImageRotation.rotation0deg;
    case 90:
      return InputImageRotation.rotation90deg;
    case 180:
      return InputImageRotation.rotation180deg;
    default:
      assert(rotation == 270);
      return InputImageRotation.rotation270deg;
  }
}

Float32List imageToByteListFloat32(
    img_lib.Image image, int inputSize, double mean, double std) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      buffer[pixelIndex++] = (pixel.r - mean) / std;
      buffer[pixelIndex++] = (pixel.g - mean) / std;
      buffer[pixelIndex++] = (pixel.b - mean) / std;
    }
  }
  return convertedBytes.buffer.asFloat32List();
}

double euclideanDistance(List e1, List e2) {
  double sum = 0.0;
  for (int i = 0; i < e1.length; i++) {
    sum += pow((e1[i] - e2[i]), 2);
  }
  return sqrt(sum);
}
