import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:facial_recognition/utils.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as img_lib;
import 'package:path_provider/path_provider.dart';
import 'package:quiver/collection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'face_mask_painters.dart';

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  ScannerScreenState createState() => ScannerScreenState();
}

class ScannerScreenState extends State<ScannerScreen> {
  static const _faceRecognitionThreshold = 1.0;
  static const _modelAssetPath = 'assets/mobile_face_net.tflite';
  static const _antiSpoofingModelPath = 'assets/face_anti_spoofing.tflite';
  static const _embeddingsFileName = 'emb.json';
  static const _cameraResolution = ResolutionPreset.high;

  static const _inputImageSize = 256;
  static const _livenessThreshold = 0.2;
  static const _laplaceThreshold = 50;
  static const _laplacianThreshold = 1000;
  static const _routeIndex = 6;

  File? _faceEmbeddingsFile;
  Multimap<String, Face>? _scanResults;
  late Interpreter _tfliteInterpreter;
  CameraController? _cameraController;
  bool _isDetecting = false;
  CameraLensDirection _direction = CameraLensDirection.front;
  Map<String, dynamic> _faceEmbeddingsMap = {};
  List<double>? _currentFaceEmbedding;
  bool _faceFound = false;
  bool _isLive = false;
  final _nameController = TextEditingController();
  late final Interpreter _antiSpoofingInterpreter;
  Directory? _tempDir;
  RootIsolateToken? _rootIsolateToken;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    await _loadModels();
    await _initializeCamera();
    await _loadFaceEmbeddings();
  }

  Future<void> _loadModels() async {
    try {
      final options = InterpreterOptions()
        ..threads = (Platform.numberOfProcessors / 2).ceil()
        ..useNnApiForAndroid = Platform.isAndroid
        ..useMetalDelegateForIOS = Platform.isIOS;

      await Future.wait([
        _loadFaceRecognitionModel(options),
        _loadAntiSpoofingModel(options),
      ]);
    } catch (e) {
      debugPrint("❌ Failed to load models: $e");
    }
  }

  Future<void> _loadFaceRecognitionModel(InterpreterOptions options) async {
    _tfliteInterpreter =
        await Interpreter.fromAsset(_modelAssetPath, options: options);
    _tfliteInterpreter.allocateTensors();
  }

  Future<void> _loadAntiSpoofingModel(InterpreterOptions options) async {
    _antiSpoofingInterpreter =
        await Interpreter.fromAsset(_antiSpoofingModelPath, options: options);
    _antiSpoofingInterpreter.allocateTensors();
  }

  Future<Multimap<String, Face>> _processFaces(
      List<Face> faces, img_lib.Image convertedImage) async {
    final results = Multimap<String, Face>();

    for (final face in faces) {
      final croppedFace = _cropAndResizeFace(convertedImage, face);

      final laplacianScore = _calculateLaplacian(croppedFace);
      debugPrint('Laplacian score: $laplacianScore');

      if (laplacianScore < _laplacianThreshold) {
        if (mounted) {
          setState(() => _isLive = false);
        }
        results.add("LOW QUALITY", face);
        continue;
      }

      final isLive = await _checkLiveness(croppedFace);
      if (mounted) {
        setState(() => _isLive = isLive);
      }

      if (!isLive) {
        results.add("SPOOF", face);
        continue;
      }

      final embedding = await Utils.recogniseInIsolate(
        croppedFace,
        _tfliteInterpreter,
        _rootIsolateToken!,
      );

      _currentFaceEmbedding = List.from(embedding);
      final label = _compare(embedding).toUpperCase();
      results.add(label, face);
    }

    return results;
  }

  Future<bool> _checkLiveness(img_lib.Image image) async {
    try {
      final input = _normalizeImage(image)
          .reshape([1, _inputImageSize, _inputImageSize, 3]);

      final clssPred = List.filled(1 * 8, 0.0).reshape([1, 8]);
      final leafNodeMask = List.filled(1 * 8, 0.0).reshape([1, 8]);

      final outputs = {
        0: clssPred,
        1: leafNodeMask,
      };

      _antiSpoofingInterpreter.runForMultipleInputs([input], outputs);

      debugPrint('Class predictions: ${clssPred[0]}');
      debugPrint('Leaf node masks: ${leafNodeMask[0]}');

      final score = _leafScore1(clssPred, leafNodeMask);
      debugPrint('Anti-spoofing score: $score');

      final routeScore = _leafScore2(clssPred);
      debugPrint('Route index score: $routeScore');

      return score > _livenessThreshold;
    } catch (e) {
      debugPrint('Error in liveness check: $e');
      return false;
    }
  }

  double _leafScore1(List clssPred, List leafNodeMask) {
    double score = 0;
    for (int i = 0; i < 8; i++) {
      score +=
          (clssPred[0][i] as double).abs() * (leafNodeMask[0][i] as double);
    }
    return score;
  }

  double _leafScore2(List clssPred) {
    return clssPred[0][_routeIndex] as double;
  }

  Float32List _normalizeImage(img_lib.Image image) {
    final img = img_lib.copyResizeCropSquare(image, size: _inputImageSize);
    final buffer = Float32List(1 * _inputImageSize * _inputImageSize * 3);
    int pixelIndex = 0;

    for (int i = 0; i < _inputImageSize; i++) {
      for (int j = 0; j < _inputImageSize; j++) {
        final pixel = img.getPixel(j, i);

        buffer[pixelIndex++] = (pixel.r / 255.0);
        buffer[pixelIndex++] = (pixel.g / 255.0);
        buffer[pixelIndex++] = (pixel.b / 255.0);
      }
    }

    return buffer;
  }

  int _calculateLaplacian(img_lib.Image image) {
    const laplaceKernel = [
      [0, 1, 0],
      [1, -4, 1],
      [0, 1, 0]
    ];

    final img = img_lib.copyResizeCropSquare(image, size: _inputImageSize);
    final greyImg = img_lib.grayscale(img);
    int score = 0;

    for (int x = 0; x < _inputImageSize - laplaceKernel.length + 1; x++) {
      for (int y = 0; y < _inputImageSize - laplaceKernel.length + 1; y++) {
        int result = 0;
        for (int i = 0; i < laplaceKernel.length; i++) {
          for (int j = 0; j < laplaceKernel.length; j++) {
            result += (greyImg.getPixel(x + i, y + j).r * laplaceKernel[i][j])
                .toInt();
          }
        }
        if (result > _laplaceThreshold) {
          score++;
        }
      }
    }

    return score;
  }

  String _compare(List<double> currentEmbedding) {
    if (_faceEmbeddingsMap.isEmpty) return "No Face saved";

    double minDist = double.infinity;
    String predictedLabel = "NOT RECOGNIZED";

    _faceEmbeddingsMap.forEach((label, savedEmbedding) {
      final distance =
          Utils.euclideanDistance(savedEmbedding, currentEmbedding);
      if (distance <= _faceRecognitionThreshold && distance < minDist) {
        minDist = distance;
        predictedLabel = label;
      }
    });

    return predictedLabel;
  }

  img_lib.Image _cropAndResizeFace(img_lib.Image image, Face face) {
    final rect = face.boundingBox;
    final x = (rect.left - 10).round();
    final y = (rect.top - 10).round();
    final w = (rect.width + 10).round();
    final h = (rect.height + 10).round();

    return img_lib.copyResizeCropSquare(
      img_lib.copyCrop(image, x: x, y: y, width: w, height: h),
      size: 112,
    );
  }

  Future<void> _loadFaceEmbeddings() async {
    _tempDir ??= await getApplicationDocumentsDirectory();
    _faceEmbeddingsFile = File('${_tempDir!.path}/$_embeddingsFileName');

    if (_faceEmbeddingsFile!.existsSync()) {
      try {
        final content = await _faceEmbeddingsFile!.readAsString();
        _faceEmbeddingsMap = json.decode(content);
      } catch (e) {
        debugPrint("Error loading face embeddings: $e");
      }
    }
  }

  Future<void> _initializeCamera() async {
    try {
      final description = await Utils.getCamera(_direction);
      Utils.rotationIntToImageRotation(description.sensorOrientation);

      await _cameraController?.dispose();

      _cameraController = CameraController(
        description,
        _cameraResolution,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      await Future.delayed(const Duration(milliseconds: 300));

      _rootIsolateToken = RootIsolateToken.instance;

      _cameraController!.startImageStream(_processCameraImage);
    } catch (e) {
      debugPrint("Camera initialization error: $e");
      await Future.delayed(const Duration(seconds: 1));
      await _initializeCamera();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Recognition')),
      body: _buildCameraPreview(),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            backgroundColor:
                (_faceFound && _isLive) ? Colors.blue : Colors.blueGrey,
            onPressed: _faceFound && _isLive ? _addLabel : null,
            child: const Icon(Icons.add),
          ),
          const SizedBox(height: 10),
          FloatingActionButton(
            onPressed: _toggleCameraDirection,
            child: Icon(_direction == CameraLensDirection.back
                ? Icons.camera_front
                : Icons.camera_rear),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(_cameraController!),
        if (_scanResults != null && _cameraController!.value.isInitialized)
          CustomPaint(
            painter: FaceDetectorPainter(
                Size(
                  _cameraController!.value.previewSize!.height,
                  _cameraController!.value.previewSize!.width,
                ),
                _scanResults,
                _isLive),
          ),
      ],
    );
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isDetecting || _cameraController == null) return;

    _isDetecting = true;
    try {
      final rotation = Utils.rotationIntToImageRotation(
        _cameraController!.description.sensorOrientation,
      );

      final inputImage = await Utils.convertCameraImageToInputImage(
        image,
        rotation,
        _cameraController!,
        _rootIsolateToken!,
      );

      if (inputImage == null) return;

      final faces = await Utils.runFaceDetectionInIsolate(
        inputImage,
        _rootIsolateToken!,
      );

      _faceFound = faces.isNotEmpty;
      final convertedImage = await Utils.convertCameraImageInIsolate(
        image,
        _direction,
        _rootIsolateToken!,
      );

      final results = await _processFaces(faces, convertedImage);

      if (mounted) {
        setState(() => _scanResults = results);
      }
    } catch (e) {
      debugPrint("Image processing error: $e");
    } finally {
      _isDetecting = false;
    }
  }

  Future<void> _toggleCameraDirection() async {
    _direction = _direction == CameraLensDirection.back
        ? CameraLensDirection.front
        : CameraLensDirection.back;

    await _disposeCamera();
    await _initializeCamera();
  }

  Future<void> _disposeCamera() async {
    if (_cameraController?.value.isStreamingImages ?? false) {
      await _cameraController?.stopImageStream();
      await _cameraController?.dispose();
      _cameraController = null;
    }
  }

  void _addLabel() {
    setState(() => _cameraController = null);

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Add Face"),
        content: Form(
          key: GlobalKey<FormState>(),
          child: TextFormField(
            controller: _nameController,
            validator: (value) =>
                value?.trim().isEmpty ?? true ? "Name cannot be empty" : null,
            decoration: const InputDecoration(
              labelText: "Name",
              icon: Icon(Icons.face),
            ),
          ),
        ),
        actions: [
          TextButton(
            child: const Text("Save"),
            onPressed: () =>
                _saveFace(_nameController.text.trim().toUpperCase()),
          ),
          TextButton(
            child: const Text("Cancel"),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
    );
  }

  Future<void> _saveFace(String name) async {
    if (name.isEmpty) return;

    _faceEmbeddingsMap[name] = _currentFaceEmbedding;
    await _faceEmbeddingsFile!.writeAsString(json.encode(_faceEmbeddingsMap));

    if (mounted) {
      Navigator.pop(context);
      await _initializeCamera();
    }

    _nameController.clear();
  }
}
