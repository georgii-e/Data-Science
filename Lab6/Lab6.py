import cv2
import torch
import time
from ultralytics import YOLO

nano_model = YOLO('yolov8n.pt')
medium_model = YOLO('yolov8m.pt')
xtra_model = YOLO('yolov8x.pt')
segmentation_model = YOLO('yolov8x-seg.pt')
torch.cuda.set_device(0)

VIDEO_FILE = r"C:/Users/egorv/PycharmProjects/pythonProject/Data science/Lab5/street.mp4"


def obj_detection_video(path, model):
    """
    Detects objects on the video

    Parameters
    ----------
    path: path to the video file
    model: model that will be used
    """
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    # Loop through the video frames
    start = time.time()
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True,  #device='cpu',
                                  conf=0.2)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            result.write(annotated_frame)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    end = time.time()
    print(f"Total taken {end - start} seconds")
    result.release()
    cap.release()
    cv2.destroyAllWindows()


print('Choose desired model:')
print('1 - Nano')
print('2 - Medium')
print('3 - Xtra Large')
print('4 - Segmentation')
mode = int(input('Mode:'))

if mode == 1:
    print('1 - Nano (press q to close the window)')
    obj_detection_video(VIDEO_FILE, nano_model)

if mode == 2:
    print('2 - Medium (press q to close the window)')
    obj_detection_video(VIDEO_FILE, medium_model)

if mode == 3:
    print('3 - Xtra Large (press q to close the window)')
    obj_detection_video(VIDEO_FILE, xtra_model)

if mode == 4:
    print('4 - Segmentation (press q to close the window)')
    obj_detection_video(VIDEO_FILE, segmentation_model)
