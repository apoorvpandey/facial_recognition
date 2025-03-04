import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as img_lib;
import 'package:screen_brightness/screen_brightness.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

class Utils {
  static Future<CameraDescription> getCamera(CameraLensDirection dir) async {
    return await availableCameras().then(
      (List<CameraDescription> cameras) => cameras.firstWhere(
        (CameraDescription camera) => camera.lensDirection == dir,
      ),
    );
  }

  static InputImageRotation rotationIntToImageRotation(int rotation) {
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

  static Future<img_lib.Image> convertCameraImageInIsolate(CameraImage image,
      CameraLensDirection dir, RootIsolateToken rootIsolateToken) async {
    return await Isolate.run<img_lib.Image>(() {
      // Initialize BackgroundIsolateBinaryMessenger
      BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);

      int width = image.width;
      int height = image.height;
      var img = img_lib.Image(width: width, height: height);

      final int uvyButtonStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel!;

      for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
          final int uvIndex = uvPixelStride * (x / 2).floor() +
              uvyButtonStride * (y / 2).floor();
          final int index = y * width + x;
          final yp = image.planes[0].bytes[index];
          final up = image.planes[1].bytes[uvIndex];
          final vp = image.planes[2].bytes[uvIndex];
          int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
          int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
              .round()
              .clamp(0, 255);
          int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

          img.setPixelRgba(x, y, r, g, b, 255);
        }
      }

      var img1 = (dir == CameraLensDirection.front)
          ? img_lib.copyRotate(img, angle: -90)
          : img_lib.copyRotate(img, angle: 90);
      return img1;
    });
  }

  static Float32List imageToByteListFloat32(
      img_lib.Image image, int inputSize, double mean, double std) {
    final int len = inputSize * inputSize * 3;
    final Uint8List byteData = Uint8List(len);
    final Float32List floatData = Float32List(len);

    int index = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        byteData[index++] = pixel.r.toInt();
        byteData[index++] = pixel.g.toInt();
        byteData[index++] = pixel.b.toInt();
      }
    }

    for (int i = 0; i < len; i++) {
      floatData[i] = (byteData[i] - mean) / std;
    }

    return floatData;
  }

  static double euclideanDistance(List e1, List e2) {
    double sum = 0.0;
    for (int i = 0; i < e1.length; i++) {
      sum += pow((e1[i] - e2[i]), 2);
    }
    return sqrt(sum);
  }

  static Future<InputImage?> convertCameraImageToInputImage(
      CameraImage cameraImage,
      InputImageRotation imageRotation,
      CameraController cameraController,
      RootIsolateToken rootIsolateToken) async {
    return await Isolate.run<InputImage?>(() {
      // Initialize BackgroundIsolateBinaryMessenger
      BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);

      try {
        final InputImageFormat format = Platform.isAndroid
            ? InputImageFormat.nv21
            : InputImageFormat.bgra8888;

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

        if (cameraImage.planes.length != 1) {
          if (kDebugMode) {
            print(
                "❌ Unsupported number of planes: ${cameraImage.planes.length}");
          }
          return null;
        }

        final Plane plane = cameraImage.planes.first;
        return InputImage.fromBytes(
          bytes: plane.bytes,
          metadata: InputImageMetadata(
            size: Size(
                cameraImage.width.toDouble(), cameraImage.height.toDouble()),
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
    });
  }

  static Future<List<double>> recogniseInIsolate(img_lib.Image img,
      Interpreter interpreter, RootIsolateToken rootIsolateToken) async {
    return await Isolate.run<List<double>>(() {
      // Initialize BackgroundIsolateBinaryMessenger
      BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);

      List input = Utils.imageToByteListFloat32(img, 112, 128, 128);
      input = input.reshape([1, 112, 112, 3]);
      List output = List.filled(1 * 192, 0.0).reshape([1, 192]);
      interpreter.run(input, output);
      output = output.reshape([192]);
      return List.from(output);
    });
  }

  static Future<List<Face>> runFaceDetectionInIsolate(
      InputImage inputImage, RootIsolateToken rootIsolateToken) async {
    return await Isolate.run<List<Face>>(() async {
      // Initialize BackgroundIsolateBinaryMessenger
      BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);

      final FaceDetector faceDetector = FaceDetector(
        options: FaceDetectorOptions(
          enableContours: true,
          enableLandmarks: true,
          enableTracking: true,
          enableClassification: true,
        ),
      );

      final faces = await faceDetector.processImage(inputImage);
      await faceDetector.close();
      return faces;
    });
  }

  static Future<void> setBrightnessToMaxAndEnableWakeLock() async {
    try {
      await ScreenBrightness.instance.setSystemScreenBrightness(1.0);
      WakelockPlus.toggle(enable: true);
    } catch (e) {
      debugPrint(e.toString());
      throw 'Failed to set system brightness';
    }
  }

  static Future<void> disableWakeLock() async {
    bool wakelockEnabled = await WakelockPlus.enabled;
    if (wakelockEnabled) {
      WakelockPlus.toggle(enable: false);
    }
  }
}
