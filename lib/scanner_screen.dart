import 'dart:convert';
import 'dart:io'; // Import for File class
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:facial_recognition/utils.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as img;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'face_mask_painters.dart'; // Use 'img' as alias for the image package

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<StatefulWidget> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  dynamic _scanResults;
  CameraController? _camera;
  bool _isDetecting = false;
  bool shouldShowDialog = false;
  bool _isEyesClosed = false;
  CameraLensDirection _direction = CameraLensDirection.front;

  late Interpreter _tfliteInterpreter; // TFLite interpreter
  bool _isModelLoaded = false;

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
        enableContours: true,
        enableLandmarks: true,
        enableClassification: true,
        // Enable classification for eye probabilities
        performanceMode:
            FaceDetectorMode.accurate // Use accurate mode for better results
        ),
  );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Recognition')),
      body: _camera == null
          ? const Center(child: CircularProgressIndicator())
          : _buildImage(),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          ElevatedButton(
            onPressed: _registerFace,
            child: const Text('Register Face'),
          ),
          FloatingActionButton(
            onPressed: _toggleCameraDirection,
            child: _direction == CameraLensDirection.back
                ? const Icon(Icons.camera_front)
                : const Icon(Icons.camera_rear),
          ),
          ElevatedButton(
            onPressed: _verifyFace,
            child: const Text('Verify Face'),
          ),
        ],
      ),
    );
  }

  Widget _buildImage() {
    return Container(
      constraints: const BoxConstraints.expand(),
      child: Stack(
        fit: StackFit.expand,
        children: <Widget>[
          CameraPreview(_camera!),
          // if (_scanResults != null) _buildResults(),
        ],
      ),
    );
  }

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadTFLiteModel();
  }

  Future<void> _loadTFLiteModel() async {
    try {
      _tfliteInterpreter = await Interpreter.fromAsset('assets/facenet.tflite');
      _tfliteInterpreter.allocateTensors();
      setState(() {
        _isModelLoaded = true;
      });
      if (kDebugMode) {
        print("✅ TFLite model loaded successfully!");
      }
    } catch (e) {
      if (kDebugMode) {
        print("❌ Failed to load TFLite model: $e");
      }
    }
  }

  Future<void> _initializeCamera() async {
    final CameraDescription cameraDescription =
        await ScannerUtils.getCamera(_direction);

    _camera = CameraController(cameraDescription, ResolutionPreset.max);

    await _camera?.initialize();

    await _camera?.startImageStream((CameraImage image) {
      if (_isDetecting) return;
      _isDetecting = true;
      final sensorOrientation = cameraDescription.sensorOrientation;
      ScannerUtils.detect(
        image: image,
        detectInImage: _faceDetector.processImage,
        imageRotation:
            InputImageRotationValue.fromRawValue(sensorOrientation) ??
                InputImageRotation.rotation0deg,
        cameraController: _camera!,
      ).then(
        (dynamic results) {
          setState(() {
            _scanResults = results;
          });

          // Check for blink
          if (kDebugMode) {
            print('kkkkkk: $_scanResults');
          }
          if (_scanResults != null && _scanResults is List<Face>) {
            for (Face face in _scanResults) {
              _checkForBlink(face);
            }
          }
        },
      ).whenComplete(() => Future.delayed(
            const Duration(milliseconds: 100),
            () => _isDetecting = false,
          ));
    });
  }

  void _checkForBlink(Face face) {
    if (kDebugMode) {
      print('kkkkkk: ${face.rightEyeOpenProbability}');
    }
    if (kDebugMode) {
      print('kkkkkk: ${face.leftEyeOpenProbability}');
    }
    final double leftEyeOpenProbability = face.leftEyeOpenProbability ?? 0.0;
    final double rightEyeOpenProbability = face.rightEyeOpenProbability ?? 0.0;
    // Detect if eyes are closed
    if (leftEyeOpenProbability < 0.3 && rightEyeOpenProbability < 0.3) {
      if (!_isEyesClosed) {
        _isEyesClosed = true;
        _handleEyesClosed();
      }
    } else {
      _isEyesClosed = false;
    }
  }

  void _handleEyesClosed() {
    if (!shouldShowDialog) {
      setState(() {
        shouldShowDialog = true;
      });
      _verifyFace();
    }
  }

  Future<void> _registerFace() async {
    if (_camera == null) {
      if (kDebugMode) {
        print('❌ Camera not initialized');
      }
      return;
    }

    final XFile imageFile = await _camera!.takePicture();
    final List<double> faceEmbedding =
        await _extractFaceEmbedding(imageFile.path);

    if (faceEmbedding.isEmpty) {
      if (kDebugMode) {
        print("❌ No face detected!");
      }
      _showResultDialog(
          'No Face Detected', 'No face was detected in the image.');
      return;
    }

    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString("registered_face", jsonEncode(faceEmbedding));

    if (kDebugMode) {
      print("✅ Face Registered Successfully!");
    }
    _showResultDialog(
        'Face Registered!', 'Face has been registered successfully.');
  }

  Future<void> _verifyFace() async {
    if (_camera == null) {
      if (kDebugMode) print('❌ Camera not initialized');
      return;
    }

    if (_camera!.value.isStreamingImages) {
      await _camera!.stopImageStream();
    }

    Future.delayed(const Duration(milliseconds: 1500), () async {
      final XFile imageFile = await _camera!.takePicture();
      final List<double> newEmbedding =
          await _extractFaceEmbedding(imageFile.path);

      if (newEmbedding.isEmpty) {
        if (kDebugMode) print("❌ No face detected!");
        _showResultDialog(
            'No Face Detected', 'Please ensure your face is visible.');
        return;
      }

      SharedPreferences prefs = await SharedPreferences.getInstance();
      String? savedFaceData = prefs.getString("registered_face");

      if (savedFaceData == null) {
        if (kDebugMode) print("❌ No registered face found!");
        _showResultDialog(
            'No Registered Face', 'Please register your face first.');
        return;
      }

      List<double> savedEmbedding =
          List<double>.from(jsonDecode(savedFaceData));
      double similarity = 100 * _cosineSimilarity(newEmbedding, savedEmbedding);

      if (kDebugMode) print("Similarity Score: $similarity");

      if (similarity.round() >= 85) {
        _showResultDialog(
            'Face Matched!', 'Welcome! ${similarity.round()}% Match.');
      } else {
        _showResultDialog('Face Not Recognized',
            'Mismatch detected. ${similarity.round()}% Similarity.');
      }

      _initializeCamera();
    });
  }

  Future<List<double>> _extractFaceEmbedding(String imagePath) async {
    if (!_isModelLoaded) {
      if (kDebugMode) print("❌ TFLite model not loaded!");
      return [];
    }

    // Load the image
    final image = img.decodeImage(File(imagePath).readAsBytesSync())!;

    // 🔹 Step 1: Run Google ML Kit's Face Detector
    final inputImage = InputImage.fromFilePath(imagePath);
    final List<Face> faces = await _faceDetector.processImage(inputImage);

    if (faces.isEmpty) {
      if (kDebugMode) print("❌ No face detected in image!");
      return [];
    }

    // 🔹 Step 2: Crop the detected face (first face found)
    final Face face = faces.first;
    final rect = face.boundingBox;

    final croppedFace = img.copyCrop(
      image,
      x: rect.left.toInt(),
      y: rect.top.toInt(),
      width: rect.width.toInt(),
      height: rect.height.toInt(),
    );

    // 🔹 Step 3: Resize, Normalize, and Extract Embeddings
    final resizedImage = img.copyResize(croppedFace, width: 112, height: 112);
    final input = imageToFloat32List(resizedImage, 112, 127.5, 128.0);

    // Run inference
    final output = List.filled(128, 0.0).reshape([1, 128]);
    _tfliteInterpreter.run(input.reshape([1, 112, 112, 3]), output);

    return _normalize(List<double>.from(output[0]));
  }

// Cropping function
  img.Image _cropFace(img.Image image) {
    int x = (image.width * 0.2).toInt();
    int y = (image.height * 0.2).toInt();
    int w = (image.width * 0.6).toInt();
    int h = (image.height * 0.6).toInt();
    return img.copyCrop(image, x: x, y: y, width: w, height: h);
  }

  Float32List preprocessImage(img.Image image, int inputSize) {
    final resizedImage =
        img.copyResize(image, width: inputSize, height: inputSize);
    final convertedBytes = Float32List(inputSize * inputSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resizedImage.getPixel(x, y);
        final mean = 127.5, std = 128.0; // Normalize

        convertedBytes[pixelIndex] = (pixel.r - mean) / std;
        convertedBytes[pixelIndex + 1] = (pixel.g - mean) / std;
        convertedBytes[pixelIndex + 2] = (pixel.b - mean) / std;
        pixelIndex += 3;
      }
    }

    return convertedBytes;
  }

  Float32List imageToFloat32List(
      img.Image image, int inputSize, double mean, double std) {
    final convertedBytes = Float32List(inputSize * inputSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);

        // Extract red, green, and blue components from the Pixel object
        final red = pixel.r.toDouble();
        final green = pixel.g.toDouble();
        final blue = pixel.b.toDouble();

        // Normalize and store in the buffer
        convertedBytes[pixelIndex] = (red - mean) / std;
        convertedBytes[pixelIndex + 1] = (green - mean) / std;
        convertedBytes[pixelIndex + 2] = (blue - mean) / std;
        pixelIndex += 3;
      }
    }

    return convertedBytes;
  }

  List<double> _normalize(List<double> embedding) {
    double norm = sqrt(embedding.map((x) => x * x).reduce((a, b) => a + b));
    return embedding.map((x) => x / norm).toList();
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dotProduct / (sqrt(normA) * sqrt(normB));
  }

  void _showResultDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(title),
          content: Text(message, style: const TextStyle(fontSize: 18)),
          actions: <Widget>[
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                setState(() {
                  shouldShowDialog = false; // Reset the flag
                });
              },
              child: const Text('OK'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _toggleCameraDirection() async {
    _direction = _direction == CameraLensDirection.back
        ? CameraLensDirection.front
        : CameraLensDirection.back;

    await _camera?.stopImageStream();
    await _camera?.dispose();

    setState(() {
      _camera = null;
    });

    await _initializeCamera();
  }

  Widget _buildResults() {
    if (_scanResults == null ||
        _camera == null ||
        !_camera!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    final Size imageSize = Size(
      _camera!.value.previewSize!.height,
      _camera!.value.previewSize!.width,
    );

    return CustomPaint(
      painter: FaceMask(imageSize, _scanResults, _handleEyesClosed),
    );
  }

  @override
  void dispose() {
    _camera?.dispose();
    _tfliteInterpreter.close();
    super.dispose();
  }
}
