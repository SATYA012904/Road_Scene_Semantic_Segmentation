
from pathlib import Path
from uuid import uuid4
from time import perf_counter

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO


# --------------------------------------------------
# 1. Flask configuration
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# --------------------------------------------------
# 2. Load models once when Flask starts
# --------------------------------------------------

print("Loading U-Net...")
unet_model = tf.keras.models.load_model(
    MODEL_DIR / "best_unet_model.keras",
    compile=False
)
print("U-Net loaded successfully!")

print("Loading DeepLabV3+...")
deeplab_model = tf.saved_model.load(
    str(MODEL_DIR / "deeplabv3plus_savedmodel")
)
deeplab_infer = deeplab_model.signatures["serve"]
print("DeepLabV3+ loaded successfully!")

print("Loading YOLOv8n...")
yolo_model = YOLO(str(MODEL_DIR / "yolov8n.pt"))
print("YOLOv8n loaded successfully!")


# --------------------------------------------------
# 3. Cityscapes class names and colors
# --------------------------------------------------

CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "Traffic Light", "Traffic Sign", "Vegetation",
    "Terrain", "Sky", "Person", "Rider", "Car",
    "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
]

CITYSCAPES_COLORS = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]
], dtype=np.uint8)


# --------------------------------------------------
# 4. Helper functions
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(image):
    resized = cv2.resize(
        image,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    batch = np.expand_dims(rgb_image.astype(np.float32), axis=0)

    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    return batch


def colorize_mask(mask):
    return CITYSCAPES_COLORS[mask]


# --------------------------------------------------
# 5. Main dashboard
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# 6. Image analysis API
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze_image():

    start_time = perf_counter()

    try:
        # Validate uploaded file
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "Only JPG, JPEG and PNG images are supported currently."
            }), 400

        # Read selected options
        model_name = request.form.get("model", "unet").lower()
        use_yolo = request.form.get("use_yolo", "true").lower() == "true"

        if model_name not in {"unet", "deeplab"}:
            return jsonify({"error": "Invalid segmentation model."}), 400

        # Generate unique filename
        extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        unique_name = f"{uuid4().hex}.{extension}"

        upload_path = UPLOAD_DIR / unique_name

        file.save(upload_path)

        # Read image
        original = cv2.imread(str(upload_path))

        if original is None:
            return jsonify({"error": "Unable to read uploaded image."}), 400

        original_height, original_width = original.shape[:2]

        # Preprocess
        batch = preprocess_image(original)

        # Run selected segmentation model
        if model_name == "unet":
            prediction = unet_model.predict(batch, verbose=0)
        else:
            prediction = deeplab_infer(
                input_layer_1=tf.convert_to_tensor(batch)
            )["output_0"].numpy()

        # Convert probabilities to class IDs
        mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)

        # Resize mask to original image dimensions
        mask = cv2.resize(
            mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Generate colored segmentation output
        colored_mask = colorize_mask(mask)

        # Convert RGB color mask to BGR for OpenCV
        colored_mask_bgr = cv2.cvtColor(
            colored_mask,
            cv2.COLOR_RGB2BGR
        )

        # Blend original image and segmentation mask
        overlay = cv2.addWeighted(
            original, 0.5,
            colored_mask_bgr, 0.5,
            0
        )

        # Save segmentation output
        segmentation_name = f"seg_{unique_name.rsplit('.', 1)[0]}.png"
        segmentation_path = RESULT_DIR / segmentation_name

        cv2.imwrite(str(segmentation_path), overlay)

        # Class distribution
        pixel_counts = np.bincount(
            mask.flatten(),
            minlength=len(CLASS_NAMES)
        )

        total_pixels = mask.size

        class_distribution = []

        for class_id, class_name in enumerate(CLASS_NAMES):

            count = int(pixel_counts[class_id])

            if count > 0:
                percentage = (count / total_pixels) * 100

                class_distribution.append({
                    "class": class_name,
                    "pixels": count,
                    "percentage": round(percentage, 2)
                })

        # YOLO object detection
        object_counts = {}
        detection_url = None
        total_objects = 0

        if use_yolo:

            results = yolo_model.predict(
                source=original,
                conf=0.25,
                iou=0.45,
                verbose=False
            )

            result = results[0]

            for class_id, count in result.names.items():
                object_counts[count] = 0

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]

                object_counts[class_name] = (
                    object_counts.get(class_name, 0) + 1
                )

            total_objects = sum(object_counts.values())

            detection_name = f"det_{unique_name.rsplit('.', 1)[0]}.jpg"
            detection_path = RESULT_DIR / detection_name

            cv2.imwrite(str(detection_path), result.plot())

            detection_url = url_for(
                "static",
                filename=f"results/{detection_name}"
            )

        # Processing time
        processing_time = round(
            perf_counter() - start_time,
            3
        )

        # Return results to frontend
        return jsonify({
            "success": True,
            "model": model_name,
            "original": url_for(
                "static",
                filename=f"uploads/{unique_name}"
            ),
            "segmentation": url_for(
                "static",
                filename=f"results/{segmentation_name}"
            ),
            "detection": detection_url,
            "total_objects": total_objects,
            "object_counts": object_counts,
            "class_distribution": class_distribution,
            "processing_time": processing_time,
            "image_width": original_width,
            "image_height": original_height
        })

    except Exception as error:
        app.logger.exception("Image analysis failed")
        return jsonify({
            "error": str(error)
        }), 500


# --------------------------------------------------
# 7. Run Flask
# --------------------------------------------------
@app.route("/api/analyze-video", methods=["POST"])
def analyze_video():

    start_time = perf_counter()

    cap = None
    writer = None

    try:
        # Validate uploaded video
        if "file" not in request.files:
            return jsonify({"error": "No video uploaded."}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No video selected."}), 400

        extension = file.filename.rsplit(".", 1)[-1].lower()

        if extension not in {"mp4", "avi", "mov"}:
            return jsonify({
                "error": "Only MP4, AVI and MOV videos are supported."
            }), 400

        # Read selected options
        model_name = request.form.get("model", "unet").lower()
        use_yolo = request.form.get("use_yolo", "true").lower() == "true"

        if model_name not in {"unet", "deeplab"}:
            return jsonify({"error": "Invalid segmentation model."}), 400

        # Save uploaded video
        unique_id = uuid4().hex

        input_name = f"{unique_id}.{extension}"
        output_name = f"processed_{unique_id}.mp4"

        input_path = UPLOAD_DIR / input_name
        output_path = RESULT_DIR / output_name

        file.save(input_path)

        # Open video
        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            return jsonify({"error": "Unable to open uploaded video."}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            return jsonify({"error": "Invalid video dimensions."}), 400

        # Initialize MP4 writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            return jsonify({"error": "Unable to create output video."}), 500

        frame_count = 0
        detection_counts = {}

        print(f"Starting video processing: {total_frames} frames")

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Preprocess frame for segmentation
            batch = preprocess_image(frame)

            # Run selected segmentation model
            if model_name == "unet":
                prediction = unet_model.predict(batch, verbose=0)
            else:
                prediction = deeplab_infer(
                    input_layer_1=tf.convert_to_tensor(batch)
                )["output_0"].numpy()

            # Generate predicted class mask
            mask = np.argmax(
                prediction[0],
                axis=-1
            ).astype(np.uint8)

            # Resize mask to original video dimensions
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

            # Colorize segmentation mask
            colored_mask = colorize_mask(mask)

            colored_mask_bgr = cv2.cvtColor(
                colored_mask,
                cv2.COLOR_RGB2BGR
            )

            # Create segmentation overlay
            segmented_frame = cv2.addWeighted(
                frame, 0.55,
                colored_mask_bgr, 0.45,
                0
            )

            # Run YOLO detection
            if use_yolo:

                results = yolo_model.predict(
                    source=frame,
                    conf=0.25,
                    iou=0.45,
                    verbose=False
                )

                result = results[0]

                # Count detections across frames
                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]

                    detection_counts[class_name] = (
                        detection_counts.get(class_name, 0) + 1
                    )

                # Draw YOLO bounding boxes and labels
                annotated_frame = result.plot()

                # Combine segmentation with YOLO annotations
                final_frame = cv2.addWeighted(
                    segmented_frame, 0.65,
                    annotated_frame, 0.35,
                    0
                )

            else:
                final_frame = segmented_frame

            # Write processed frame
            writer.write(final_frame)

            frame_count += 1

            # Print progress in terminal
            if frame_count % 25 == 0:
                print(
                    f"Processed {frame_count}/{total_frames} frames"
                )

        # Release video resources
        cap.release()
        cap = None

        writer.release()
        writer = None

        if frame_count == 0:
            return jsonify({"error": "No readable frames found."}), 400

        processing_time = round(
            perf_counter() - start_time,
            2
        )

        total_detections = sum(detection_counts.values())

        print("Video processing completed!")
        print(f"Processed frames: {frame_count}")
        print(f"Processing time: {processing_time} seconds")

        return jsonify({
            "success": True,
            "video": url_for(
                "static",
                filename=f"results/{output_name}"
            ),
            "model": model_name,
            "frames_processed": frame_count,
            "original_fps": round(fps, 2),
            "duration_seconds": round(frame_count / fps, 2),
            "processing_time": processing_time,
            "total_detections": total_detections,
            "detection_counts": detection_counts
        })

    except Exception as error:

        app.logger.exception("Video analysis failed")

        return jsonify({
            "error": str(error)
        }), 500

    finally:

        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False
    )





