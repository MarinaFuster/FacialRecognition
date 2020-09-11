import cv2
from mtcnn.mtcnn import MTCNN


if __name__ == '__main__':
    detector = MTCNN()
    img = cv2.imread('examples/benedict_test.jpg')
    faces = detector.detect_faces(img)  # resulttw

    # Draw faces on image
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

    cv2.imshow("MTCNN-Full", img)
    # Cropped image
    cv2.imshow("MTCNN-Cut", img[y:y1, x:x1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
