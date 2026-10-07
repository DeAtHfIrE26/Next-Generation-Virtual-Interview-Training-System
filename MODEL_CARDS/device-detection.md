# Device and second-person detection (E7)

- **Purpose:** practice-integrity notices for a visible phone or a second person.
- **Method:** MediaPipe Object Detector with EfficientDet-Lite0 (COCO; Apache-2.0) restricted to "cell phone", at 2 fps on-device; face count from Face Landmarker. Server-side debounce (3 consecutive observations) and per-type policy. Coaching mode only shows notices; proctored mode ends the session at per-type limits. Replaces the prototype's YOLOv8n (AGPL-3.0).
- **Evaluation:** `device_detection` suite (precision/recall per event). **Not measured.**
- **Known risks:** phones partly visible or held below the frame (missed); dark rectangles mistaken for phones; people passing in the background.
