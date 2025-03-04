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
  File? _faceEmbeddingsFile;
  Multimap<String, Face>? _scanResults;
  late Interpreter _tfliteInterpreter;
  CameraController? _camera;
  bool _isDetecting = false;
  CameraLensDirection _direction = CameraLensDirection.front;
  Map<String, dynamic> _faceEmbeddingsMap = {};

  // _faceRecognitionThreshold
  // Stricter Matching: Lower the threshold (e.g., 0.8)
  // Lenient Matching: Increase the threshold (e.g., 1.2)
  final double _faceRecognitionThreshold = 1.0;
  Directory? _tempDir;
  List<double>? _currentFaceEmbedding;
  bool _faceFound = false;
  final _name = TextEditingController();
  bool _isLive = false;
  int _blinkCount = 0;
  Timer? _blinkTimer;
  String _headPoseFeedback = "Look straight";
  bool _isHeadMoving = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Recognition'),
      ),
      body: _buildImage(),
      floatingActionButton:
          Column(mainAxisAlignment: MainAxisAlignment.end, children: [
        FloatingActionButton(
          backgroundColor:
              (_faceFound && _isLive) ? Colors.blue : Colors.blueGrey,
          onPressed: () {
            if (_faceFound && _isLive) _addLabel();
          },
          heroTag: null,
          child: Icon(Icons.add),
        ),
        SizedBox(
          height: 10,
        ),
        FloatingActionButton(
          onPressed: _toggleCameraDirection,
          heroTag: null,
          child: _direction == CameraLensDirection.back
              ? const Icon(Icons.camera_front)
              : const Icon(Icons.camera_rear),
        ),
      ]),
    );
  }

  @override
  void initState() {
    super.initState();

    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    _initializeCamera();
  }

  Future<void> _loadModel() async {
    try {
      final int numThreads = Platform.numberOfProcessors;
      final options = InterpreterOptions()..threads = numThreads;
      _tfliteInterpreter = await Interpreter.fromAsset(
          'assets/mobile_face_net.tflite',
          options: options);

      _tfliteInterpreter.allocateTensors();

      if (kDebugMode) {
        print("✅ TFLite model loaded successfully! with $numThreads threads");
      }
    } catch (e) {
      if (kDebugMode) {
        print("❌ Failed to load TFLite model: $e");
      }
    }
  }

  Future<void> _initializeCamera() async {
    await _loadModel();
    CameraDescription description = await getCamera(_direction);

    InputImageRotation rotation = rotationIntToImageRotation(
      description.sensorOrientation,
    );

    _camera =
        CameraController(description, ResolutionPreset.max, enableAudio: false);
    await _camera!.initialize();
    await Future.delayed(Duration(milliseconds: 500));
    _tempDir = await getApplicationDocumentsDirectory();
    String embPath = '${_tempDir!.path}/emb.json';
    _faceEmbeddingsFile = File(embPath);
    if (_faceEmbeddingsFile!.existsSync()) {
      _faceEmbeddingsMap = json.decode(_faceEmbeddingsFile!.readAsStringSync());
    }

    _camera!.startImageStream((CameraImage image) {
      if (_camera != null) {
        if (_isDetecting) return;
        _isDetecting = true;
        String res;
        dynamic finalResult = Multimap<String, Face>();
        detect(image, _getDetectionMethod(), rotation, _camera).then(
          (dynamic result) async {
            if (result.length == 0) {
              _faceFound = false;
              _isLive = false;
            } else {
              _faceFound = true;
              _checkLiveStatus(result);
              _checkHeadPose(result);
            }
            Face face;
            img_lib.Image convertedImage =
                _convertCameraImage(image, _direction);
            for (face in result) {
              double x, y, w, h;
              x = (face.boundingBox.left - 10);
              y = (face.boundingBox.top - 10);
              w = (face.boundingBox.width + 10);
              h = (face.boundingBox.height + 10);
              img_lib.Image croppedImage = img_lib.copyCrop(convertedImage,
                  x: x.round(),
                  y: y.round(),
                  width: w.round(),
                  height: h.round());
              croppedImage =
                  img_lib.copyResizeCropSquare(croppedImage, size: 112);
              res = _recognise(croppedImage);
              finalResult.add(res, face);
            }
            setState(() {
              _scanResults = finalResult;
            });

            _isDetecting = false;
          },
        ).catchError(
          (error) {
            if (kDebugMode) {
              print("error: $error");
            }
            _isDetecting = false;
          },
        );
      }
    });
  }

  void _checkLiveStatus(List<Face> faces) {
    for (var face in faces) {
      if (face.leftEyeOpenProbability != null &&
          face.rightEyeOpenProbability != null) {
        if (face.leftEyeOpenProbability! < 0.3 &&
            face.rightEyeOpenProbability! < 0.3) {
          _blinkCount++;
          if (_blinkCount >= 2) {
            _isLive = true;
            _blinkTimer?.cancel();
            _blinkTimer = Timer(Duration(milliseconds: 2500), () {
              _isLive = false;
              _blinkCount = 0;
              setState(() {});
            });
          }
        }
      }
    }
  }

  void _checkHeadPose(List<Face> faces) {
    for (var face in faces) {
      double? yaw = face.headEulerAngleY; // Left/Right rotation
      double? pitch = face.headEulerAngleX; // Up/Down rotation

      if (yaw! > 15) {
        setState(() {
          _headPoseFeedback = "Turn your head left";
        });
      } else if (yaw < -15) {
        setState(() {
          _headPoseFeedback = "Turn your head right";
        });
      } else if (pitch! > 15) {
        setState(() {
          _headPoseFeedback = "Tilt your head up";
        });
      } else if (pitch < -15) {
        setState(() {
          _headPoseFeedback = "Tilt your head down";
        });
      } else {
        setState(() {
          _headPoseFeedback = "Look straight";
        });
      }

      // Check if the user has completed the head movement
      if ((yaw > 15 || yaw < -15 || pitch! > 15 || pitch < -15) &&
          !_isHeadMoving) {
        _isHeadMoving = true;
        Timer(Duration(seconds: 2), () {
          _isHeadMoving = false;
          _isLive = true; // Mark as live if head movement is detected
        });
      }
    }
  }

  HandleDetection _getDetectionMethod() {
    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.accurate,
        enableLandmarks: true,
        enableContours: true,
        enableTracking: true,
        enableClassification: true,
      ),
    );
    return faceDetector.processImage;
  }

  Widget _buildResults() {
    const Text noResultsText = Text('');
    if (_scanResults == null ||
        _camera == null ||
        !_camera!.value.isInitialized) {
      return noResultsText;
    }
    CustomPainter painter;

    final Size imageSize = Size(
      _camera!.value.previewSize!.height,
      _camera!.value.previewSize!.width,
    );
    painter = FaceDetectorPainter(
        imageSize, _scanResults, _isLive, _headPoseFeedback);
    return CustomPaint(
      painter: painter,
    );
  }

  Widget _buildImage() {
    if (_camera == null || !_camera!.value.isInitialized) {
      return Center(
        child: CircularProgressIndicator(),
      );
    }

    return Container(
      constraints: const BoxConstraints.expand(),
      child: _camera == null
          ? const SizedBox.shrink()
          : Stack(
              fit: StackFit.expand,
              children: <Widget>[
                CameraPreview(_camera!),
                _buildResults(),
              ],
            ),
    );
  }

  void _toggleCameraDirection() async {
    if (_direction == CameraLensDirection.back) {
      _direction = CameraLensDirection.front;
    } else {
      _direction = CameraLensDirection.back;
    }
    await _camera!.stopImageStream();
    await _camera!.dispose();

    setState(() {
      _camera = null;
    });

    await _initializeCamera();
  }

  img_lib.Image _convertCameraImage(
      CameraImage image, CameraLensDirection dir) {
    int width = image.width;
    int height = image.height;
    var img = img_lib.Image(width: width, height: height);

    final int uvyButtonStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        final int uvIndex =
            uvPixelStride * (x / 2).floor() + uvyButtonStride * (y / 2).floor();
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
  }

  String _recognise(img_lib.Image img) {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List.filled(1 * 192, 0.0).reshape([1, 192]);
    _tfliteInterpreter.run(input, output);
    output = output.reshape([192]);
    _currentFaceEmbedding = List.from(output);
    return _compare(_currentFaceEmbedding!).toUpperCase();
  }

  String _compare(List currEmb) {
    if (_faceEmbeddingsMap.isEmpty) return "No Face saved";
    double minDist = 999;
    String predictedResponse = "NOT RECOGNIZED";
    for (String label in _faceEmbeddingsMap.keys) {
      if (_faceEmbeddingsMap[label] == null) {
        continue;
      }
      final currDist = euclideanDistance(_faceEmbeddingsMap[label], currEmb);
      if (currDist <= _faceRecognitionThreshold && currDist < minDist) {
        minDist = currDist;
        predictedResponse = label;
      }
    }
    if (kDebugMode) {
      print("Min Distance: $minDist, Predicted: $predictedResponse");
    }
    return predictedResponse;
  }

  void _addLabel() {
    setState(() {
      _camera = null;
    });

    final formKey = GlobalKey<FormState>();

    var alert = AlertDialog(
      title: Text("Add Face"),
      content: Form(
        key: formKey,
        child: Row(
          children: <Widget>[
            Expanded(
              child: TextFormField(
                controller: _name,
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return "Name cannot be empty";
                  }
                  return null;
                },
                autofocus: true,
                decoration:
                    InputDecoration(labelText: "Name", icon: Icon(Icons.face)),
              ),
            )
          ],
        ),
      ),
      actions: <Widget>[
        TextButton(
          child: Text("Save"),
          onPressed: () async {
            if (formKey.currentState!.validate()) {
              await _handle(_name.text.trim().toUpperCase());
              _name.clear();
              if (mounted) {
                Navigator.pop(context);
              }
            }
          },
        ),
        TextButton(
          child: Text("Cancel"),
          onPressed: () async {
            await _initializeCamera();
            if (mounted) {
              Navigator.pop(context);
            }
          },
        )
      ],
    );

    showDialog(
      context: context,
      builder: (context) {
        return alert;
      },
    );
  }

  Future<void> _handle(String text) async {
    _faceEmbeddingsMap[text] = _currentFaceEmbedding;
    _faceEmbeddingsFile!.writeAsStringSync(json.encode(_faceEmbeddingsMap));
    await _initializeCamera();
  }

  @override
  void dispose() {
    _tfliteInterpreter.close();
    _blinkTimer?.cancel();
    super.dispose();
  }
}
