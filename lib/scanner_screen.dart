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
    Utils.setBrightnessToMaxAndEnableWakeLock();
  }

  Future<void> _loadModel() async {
    try {
      int numCores = Platform.numberOfProcessors;
      int numThreads = (numCores / 2).ceil();
      final options = InterpreterOptions()..threads = numThreads;

      if (Platform.isAndroid) {
        try {
          options.useNnApiForAndroid = true;
          if (kDebugMode) print("✅ NNAPI delegate enabled on Android");
        } catch (e) {
          if (kDebugMode) print("⚠️ NNAPI failed: $e");
          try {
            var gpuDelegateV2 = GpuDelegateV2(options: GpuDelegateOptionsV2());
            options.addDelegate(gpuDelegateV2);
            if (kDebugMode) print("✅ GPU delegate enabled on Android");
          } catch (e) {
            if (kDebugMode) print("⚠️ GPU delegate also failed, using CPU: $e");
          }
        }
      }

      if (Platform.isIOS) {
        try {
          options.useMetalDelegateForIOS = true;
          if (kDebugMode) print("✅ Metal delegate enabled on iOS");
        } catch (e) {
          if (kDebugMode) {
            print("⚠️ Failed to enable Metal delegate on iOS: $e");
          }
        }
      }

      _tfliteInterpreter = await Interpreter.fromAsset(
        'assets/mobile_face_net.tflite',
        options: options,
      );

      _tfliteInterpreter.allocateTensors();

      if (kDebugMode) {
        print("✅ TFLite model loaded successfully with $numThreads threads");
      }
    } catch (e) {
      if (kDebugMode) print("❌ Failed to load TFLite model: $e");
    }
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

  Future<void> _initializeCamera() async {
    await _loadModel();
    CameraDescription description = await Utils.getCamera(_direction);

    InputImageRotation rotation = Utils.rotationIntToImageRotation(
      description.sensorOrientation,
    );
    if (_camera != null) {
      await _camera!.dispose();
    }
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

    try {
      _camera!.startImageStream((CameraImage image) async {
        if (_camera != null) {
          if (_isDetecting) return;
          _isDetecting = true;

          try {
            dynamic finalResult = Multimap<String, Face>();

            // Get the RootIsolateToken
            RootIsolateToken rootIsolateToken = RootIsolateToken.instance!;

            // Convert CameraImage to InputImage
            final inputImage = await Utils.convertCameraImageToInputImage(
                image, rotation, _camera!, rootIsolateToken);
            if (inputImage == null) return;

            // Run face detection in an isolate
            List<Face> result = await Utils.runFaceDetectionInIsolate(
                inputImage, rootIsolateToken);

            if (result.isEmpty) {
              _faceFound = false;
              _isLive = false;
            } else {
              _faceFound = true;
              _checkLiveStatus(result);
              _checkHeadPose(result);
            }

            // Convert CameraImage to img_lib.Image in an isolate
            img_lib.Image convertedImage =
                await Utils.convertCameraImageInIsolate(
                    image, _direction, rootIsolateToken);

            for (Face face in result) {
              double x = (face.boundingBox.left - 10);
              double y = (face.boundingBox.top - 10);
              double w = (face.boundingBox.width + 10);
              double h = (face.boundingBox.height + 10);

              img_lib.Image croppedImage = img_lib.copyCrop(
                convertedImage,
                x: x.round(),
                y: y.round(),
                width: w.round(),
                height: h.round(),
              );
              croppedImage =
                  img_lib.copyResizeCropSquare(croppedImage, size: 112);

              // Run TensorFlow Lite inference in an isolate
              List<double> embedding = await Utils.recogniseInIsolate(
                  croppedImage, _tfliteInterpreter, rootIsolateToken);
              _currentFaceEmbedding = List.from(embedding);
              String res = _compare(embedding).toUpperCase();
              finalResult.add(res, face);
            }

            setState(() {
              _scanResults = finalResult;
            });
          } catch (error) {
            if (kDebugMode) print("detectFunctionError: $error");
          } finally {
            _isDetecting = false;
          }
        }
      });
    } catch (error) {
      if (kDebugMode) print("startImageStreamError: $error");
      await _initializeCamera();
    }
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

  String _compare(List currEmb) {
    if (_faceEmbeddingsMap.isEmpty) return "No Face saved";
    double minDist = 999;
    String predictedResponse = "NOT RECOGNIZED";
    for (String label in _faceEmbeddingsMap.keys) {
      if (_faceEmbeddingsMap[label] == null) {
        continue;
      }
      final currDist =
          Utils.euclideanDistance(_faceEmbeddingsMap[label], currEmb);
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

  Future<void> _disposeCamera() async {
    if (_camera!.value.isStreamingImages) {
      await _camera!.stopImageStream();
      await _camera!.dispose();
    }
  }

  @override
  void dispose() {
    _tfliteInterpreter.close();
    _blinkTimer?.cancel();
    _disposeCamera();
    Utils.disableWakeLock();
    super.dispose();
  }

  void _toggleCameraDirection() async {
    if (_direction == CameraLensDirection.back) {
      _direction = CameraLensDirection.front;
    } else {
      _direction = CameraLensDirection.back;
    }
    if (_camera!.value.isStreamingImages) {
      await _camera!.stopImageStream();
      await _camera!.dispose();
      setState(() {
        _camera = null;
      });
    }

    await _initializeCamera();
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
}
