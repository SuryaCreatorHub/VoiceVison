import gradio as gr
from ultralytics import YOLO

MODEL_NAME = "yolov8s-worldv2.pt"
CLASSES = [
    "person", "chair", "table", "sofa", "bed", "bottle", "cup", "phone", "book",
    "door", "doorway", "ceiling fan", "fan", "light fixture", "lamp", "light bulb",
    "stairs", "staircase", "hallway", "corridor", "pathway", "sidewalk", "road",
    "car", "bus", "bicycle", "motorcycle", "traffic light", "stop sign", "bench",
    "backpack", "handbag", "cabinet", "shelf", "window", "desk", "plant"
]
print(f"Loading {MODEL_NAME}; the first run downloads its weights.")
detector = YOLO(MODEL_NAME)
detector.set_classes(CLASSES)


def describe_frame(frame, previous_description):
    if frame is None:
        return "Start the camera to detect objects.", previous_description, []
    height, width = frame.shape[:2]
    small_frame = frame[::2, ::2, ::-1].copy()
    result = detector.predict(source=small_frame, device="cpu", imgsz=320, conf=0.30, verbose=False)[0]
    found, seen = [], set()
    if result.boxes is not None:
        for box in result.boxes:
            name = detector.names[int(box.cls.item())]
            x1, _, x2, _ = box.xyxy[0].tolist()
            center_x = ((x1 + x2) / 2) * 2
            position = "left" if center_x < width * 0.38 else "right" if center_x > width * 0.62 else "center"
            if (name, position) not in seen:
                seen.add((name, position))
                found.append({"name": name, "position": position})
    description = (
        "I can see " + ", ".join(f"{x['name']} on the {x['position']}" for x in found[:5]) + "."
        if found else "I do not detect a familiar object in view."
    )
    return description, description, found


def answer_scene_question(question, detections):
    if not question or not question.strip():
        return "Type a question about the live camera view."
    q = question.lower()
    if any(term in q for term in ("where should", "which way", "where do i go", "where to go", "where can i walk", "safe path", "safe route", "choose a route", "navigate me")) or ("where" in q and ("go" in q or "walk" in q)):
        return "I cannot safely choose a route or tell you where to walk. I can only describe objects and their positions in the camera view. Please use a trusted mobility aid or a trained guide for navigation."
    if not detections:
        return "I have not detected any familiar objects yet. Start the camera and try again."

    synonym_groups = [
        ("fan", "ceiling fan"),
        ("light", "lamp", "light fixture", "light bulb", "bulb"),
        ("path", "pathway", "hallway", "corridor", "doorway", "stairs", "staircase"),
    ]
    matches = []
    for item in detections:
        name = item["name"].lower()
        asked_for_group = next((group for group in synonym_groups if any(term in q for term in group)), None)
        if name in q or (asked_for_group and any(term in name for term in asked_for_group)):
            matches.append(item)
    if matches:
        locations = ", ".join(f"{x['name']} is on the {x['position']}" for x in matches[:3])
        return locations + "."

    for side in ("left", "right", "center"):
        if side in q or (side == "center" and ("middle" in q or "ahead" in q)):
            items = [x["name"] for x in detections if x["position"] == side]
            return f"On the {side}, I can see " + ", ".join(items) + "." if items else f"I do not detect a familiar object on the {side}."
    summary = ", ".join(f"{x['name']} on the {x['position']}" for x in detections[:5])
    return f"I can currently identify: {summary}. I can answer which detected objects are visible and where they appear."


browser_speech = r"""
() => {
  if (window.__voicevisionSpeechInstalled) return;
  window.__voicevisionSpeechInstalled = true;
  const spokenAt = new Map();
  let lastText = "";
  let stableSince = Date.now();
  let speaking = false;
  setInterval(() => {
    const field = document.querySelector("#live-description textarea");
    const enabled = document.querySelector("#speech-enabled input[type='checkbox']")?.checked;
    const text = field?.value?.trim();
    if (!enabled || !text || text === "Start the camera to begin.") return;
    if (text !== lastText) {
      lastText = text;
      stableSince = Date.now();
      return;
    }
    if (speaking || Date.now() - stableSince < 1200) return;
    const last = spokenAt.get(text) || 0;
    if (Date.now() - last < 30000) return;
    spokenAt.set(text, Date.now());
    if (window.speechSynthesis) {
      speaking = true;
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.onend = () => { speaking = false; stableSince = Date.now(); };
      utterance.onerror = () => { speaking = false; };
      window.speechSynthesis.speak(utterance);
    }
  }, 400);
}
"""

voice_input_js = r"""
() => {
  const allowed = document.querySelector("#voice-consent input[type='checkbox']")?.checked;
  if (!allowed) return "First check the voice recognition consent box below.";
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return "Voice input is not supported by this browser. Type your question instead.";
  return new Promise((resolve) => {
    const recognition = new Recognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    let completed = false;
    recognition.onresult = (event) => { completed = true; resolve(event.results[0][0].transcript); };
    recognition.onerror = (event) => resolve("Voice input error: " + event.error);
    recognition.onend = () => { if (!completed) resolve(""); };
    recognition.start();
  });
}
"""

speak_answer_js = r"""
(text) => {
  if (text && window.speechSynthesis) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
  }
  return [];
}
"""

css = r"""
#ask-by-voice, #ask-by-voice button {
  width: 116px !important;
  height: 116px !important;
  min-width: 116px !important;
  min-height: 116px !important;
  border-radius: 999px !important;
  white-space: normal !important;
  line-height: 1.25 !important;
}
"""

with gr.Blocks(title="VoiceVision", js=browser_speech, css=css) as demo:
    gr.Markdown("# VoiceVision\nLive object detection on your computer. Spoken descriptions are in English and show camera position. This demo is not a walking or road-crossing safety guide.")
    camera = gr.Image(label="Live camera", sources=["webcam"], type="numpy")
    answer = gr.Textbox(label="Live AI description", lines=3, value="Start the camera to begin.", elem_id="live-description")
    latest_detections = gr.State([])
    previous_description = gr.State("")
    gr.Checkbox(label="Speak live descriptions", value=True, elem_id="speech-enabled")
    camera.stream(
        describe_frame,
        inputs=[camera, previous_description],
        outputs=[answer, previous_description, latest_detections],
        time_limit=60,
        stream_every=0.8,
        concurrency_limit=1,
    )

    gr.Markdown("## Ask AI about the live scene")
    with gr.Row():
        question = gr.Textbox(label="Ask AI", placeholder="For example: Where is the fan?", scale=4)
        ask_button = gr.Button("Ask AI", variant="primary", scale=1)
        voice_button = gr.Button("🎙️ Ask by voice", scale=1, elem_id="ask-by-voice")
    gr.Checkbox(
        label="Allow browser voice recognition for my spoken question. The browser's recognition service may process the audio.",
        value=False,
        elem_id="voice-consent",
    )
    with gr.Row():
        ask_answer = gr.Textbox(label="Answer", lines=2, scale=4)
        speak_answer_button = gr.Button("Speak answer", scale=1)
    ask_button.click(answer_scene_question, inputs=[question, latest_detections], outputs=ask_answer)
    voice_button.click(fn=None, inputs=[], outputs=question, js=voice_input_js)
    speak_answer_button.click(fn=None, inputs=ask_answer, outputs=[], js=speak_answer_js)

demo.launch()


