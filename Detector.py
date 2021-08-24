import cv2

# to load some pre trained data

pre_trained_data_cars = cv2.CascadeClassifier("cars.xml")
pre_trained_data_fullbody = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Choose an image to detect faces in
#img = cv2.imread("g3.jpg")

video = cv2.VideoCapture(
    "Tesla Autopilot Dashcam Compilation 2018 Version.mp4")


while True:

    successful_frame_read, frame = video.read()
    # Must convert to grayscale
    grayScale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    car_coordinates = pre_trained_data_cars.detectMultiScale(grayScale_image)
    pedestrian_coordinates = pre_trained_data_fullbody.detectMultiScale(
        grayScale_image)

    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()

print("Code Complete")
