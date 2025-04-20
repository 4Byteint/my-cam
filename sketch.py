from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "YUV420"}
)
picam2.configure(config)
picam2.start()

start_time = time.time()
frame_count = 0
fps = 0

while True:
    frame = picam2.capture_array("main")
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count
        print(f"FPS: {fps}")
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Picamera2 FPS Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
