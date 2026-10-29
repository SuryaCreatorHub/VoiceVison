# VoiceVision

VoiceVision uses the local YOLOv8-Worldv2 open-vocabulary object detector. It can be prompted to look for common household and scene items such as people, fans, lights, doors, stairs, hallways, and pathways, then reports whether detected items appear on the left, center, or right of the camera view. Live descriptions use browser speech synthesis and avoid repeating the same sentence for 30 seconds. Ask AI answers questions about detected objects and their positions; it does not choose walking routes. Voice input is optional and requires the user to enable the consent checkbox; the browser's speech-recognition service may process that audio.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by the app and allow webcam access. On the first run, Ultralytics downloads `yolov8s-worldv2.pt`. Camera frames are processed on the computer and are not sent to a cloud AI service.

## AI model and license

- Model: Ultralytics YOLOv8-Worldv2 Small, `yolov8s-worldv2.pt`.
- Model/software license: AGPL-3.0. See https://github.com/ultralytics/ultralytics/blob/main/LICENSE.
- This project is released under AGPL-3.0.

## Safety scope

VoiceVision describes detected objects and their position in the camera view. It is a hackathon demo, not a navigation aid; do not rely on it to walk, avoid hazards, or cross roads.
