import cv2 as cv
import numpy as np

def motion():
    cap = cv.VideoCapture("/Users/adityasharma/Downloads/pdd_new/pedis/in.avi")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    _, frame1 = cap.read()
    _, frame2 = cap.read()

    while cap.isOpened():
        diff = cv.absdiff(frame1, frame2)
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour) < 900:
                continue
            (x, y, w, h) = cv.boundingRect(contour)
            cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame1, "Pedestrian ({})".format('Movement'), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow("Video", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv.waitKey(50) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    motion()
