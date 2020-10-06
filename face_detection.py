import cv2
from mtcnn.mtcnn import MTCNN
from image_resize import resizeImage


if __name__ == '__main__':
    detector = MTCNN()
    # img = cv2.imread('examples/benedict_test.jpg')
    img = cv2.imread('namaru.jpg')
    faces = detector.detect_faces(img)  # resulttw

    counter = 0
    # Draw faces on image
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        path_to_file = f"detection/person{counter}.jpg"
        cv2.imwrite(path_to_file, img[y:y1, x:x1])
        resizeImage(path_to_file)
        counter += 1

    cv2.imwrite("complete_image.jpg", img)
