import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import cv2

MARGIN = 10  # pixels
ROW_SIZE = 20  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
RECTANGLES_COLORS = [(17, 64, 170), (0, 203, 0), (253, 252, 1), (252, 115, 0), (113, 8, 170), (254, 0, 0)]
TEXT_COLOR = (0, 255, 255)

model_path = r'C:/Users/egorv/PycharmProjects/pythonProject/Data science/Lab5/efficientdet_lite2.tflite'
IMAGE_FILE = r'C:/Users/egorv/PycharmProjects/pythonProject/Data science/Lab5/dog-and-sofa.jpg'
VIDEO_FILE = r'C:/Users/egorv/PycharmProjects/pythonProject/Data science/Lab5/street2.mp4'


def visualize(image, detection_result):
    """
    Draws bounding boxes on the input image and return it

    Parameters
    ----------
    image: The input RGB image
    detection_result: The list of all "Detection" entities to be visualized

    Returns
    -------
    image: image with bounding boxes
    objects: names of objects on the photo
    """
    objects = []
    for i, detection in enumerate(detection_result.detections):
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR  # RECTANGLES_COLORS[i % len(RECTANGLES_COLORS)]
                      , 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        objects.append(category_name)
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (bbox.origin_x + MARGIN,  # + bbox.width//2
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image, objects


def obj_detection_img(path):
    """
    Detects objects on the photo

    Parameters
    ----------
    path: path to the image file
    """

    # STEP 2: Create an ObjectDetector object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.15, max_results=12)
    detector = vision.ObjectDetector.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(path)

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image, all_objects = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    print(f"There are {len(detection_result.detections)} objects on the photo:")
    print(all_objects)

    cv2.imshow("Object Detection", rgb_annotated_image)
    cv2.waitKey()


def obj_detection_video(path):
    """
    Detects objects on the video

    Parameters
    ----------
    path: path to the video file
    """
    # Load the video file
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.2, max_results=10,
                                           running_mode=VisionRunningMode.VIDEO)
    detector = vision.ObjectDetector.create_from_options(options)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    # Process each frame of the video
    start = time.time()
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Detect objects in the frame
        frame_timestamp_ms = int(1000 * frame_index / fps)
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        annotated_image, _ = visualize(frame, detection_result)
        result.write(frame)
        cv2.imshow("Object Detection", frame)
    # Close the video capture and all windows
    end = time.time()
    print(f"Total taken {end - start} seconds")
    cap.release()
    result.release()
    cv2.destroyAllWindows()


print('Object tracking: video or image?')
print('1 - Video')
print('2 - Image')
mode = int(input('Mode:'))

if mode == 1:
    print('1 - Video (press q to close the window)')
    obj_detection_video(VIDEO_FILE)

if mode == 2:
    print('2 - Image')
    obj_detection_img(IMAGE_FILE)
