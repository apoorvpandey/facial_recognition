# Facial Recognition Flutter Application

This Flutter application demonstrates real-time facial recognition using a combination of Google's
ML Kit for face detection and TensorFlow Lite for face embedding extraction. The application allows
users to register a face and later verify if a detected face matches the registered one.

## Features

- **Real-time Face Detection**: Detects faces using the device's camera.
- **Face Registration**: Captures and stores a face embedding for future verification.
- **Face Verification**: Compares a detected face with the registered face to determine a match.
- **Blink Detection**: Detects if the user's eyes are closed, which can be used as an additional
  security measure.
- **Multi-User Attendance System**: Supports verification and attendance logging for multiple users.

## Tools and Libraries Used

- **Flutter**: The framework used to build the application. (Version: **3.29.0**)
- **Google ML Kit**: Used for face detection and facial landmark detection.
- **TensorFlow Lite**: Used for extracting face embeddings from detected faces.
- **Camera Plugin**: Provides access to the device's camera for real-time image capture.
- **Shared Preferences**: Used to store the registered face embedding locally.
- **Image Package**: Used for image manipulation and preprocessing before feeding into the
  TensorFlow Lite model.

## Dependencies

```yaml
camera: ^0.10.6
google_ml_kit: ^0.19.0
tflite_flutter: ^0.11.0
shared_preferences: ^2.5.2
image: ^4.5.2
google_mlkit_face_detection: ^0.12.0
google_mlkit_commons: ^0.9.0
```

## How It Works

### 1. Initialization

- The application initializes the camera and loads the TensorFlow Lite model (`facenet.tflite`) for
  face embedding extraction.

### 2. Face Detection

- The camera stream is continuously analyzed using Google ML Kit's face detection capabilities.
- Detected faces are processed to extract facial landmarks and check for eye closure (blink
  detection).

### 3. Face Registration

- When the user registers a face, the application captures an image, extracts the face embedding
  using the TensorFlow Lite model, and stores it locally using Shared Preferences.

### 4. Face Verification

- During verification, the application captures a new image, extracts the face embedding, and
  compares it with the stored embedding using cosine similarity.
- If the similarity score is above a certain threshold (e.g., 85%), the face is considered a match.

### 5. Multi-User Attendance System

- The system allows multiple users to register and verify their attendance.
- Each detected face is checked against stored embeddings.
- If a match is found, the corresponding user’s attendance is logged.
- Attendance records can be retrieved for tracking purposes.

### 6. Blink Detection

- The application checks if the user's eyes are closed by analyzing the eye open probabilities
  provided by Google ML Kit.
- If both eyes are closed, the application triggers the face verification process.

## Known Bug

Sometimes if you blink your eyes, the app crashes, and TensorFlow Lite is unable to detect faces. To
counter this, we can handle animation during `setState`, or as a quick fix, modify the following
widget:

```dart
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
```

### Quick Fix:

Comment out the usage of this line:

```dart
// if (_scanResults != null) _buildResults(),
```

This will remove the mask overlay, but the functionality will still work.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/apoorvpandey/facial_recognition.git
cd facial_recognition
```

### 2. Install Dependencies

```bash
flutter pub get
```

### 3. Run the Application

```bash
flutter run
```

## How to Use the Facial Recognition App

### 1. Open the App

- Launch the app on your device.
- The camera will start automatically.

### 2. Register a Face

- Click on the "Register Face" button.
- The app will capture your face and process the image.
- Enter your name when prompted.
- Your face will be stored in the database.

### 3. Verify a Face

- Click on the "Verify Face" button.
- The app will scan your face and compare it with stored faces.
- If a match is found, your name will be displayed.
- If no match is found, you will be notified.

### 4. Switch Camera

- Tap the camera switch button to change between front and back cameras.

### 5. View Registered Users

- Open the "Registered Users" screen to see the list of stored faces.

### 6. Close the App

- Simply exit the app when done.