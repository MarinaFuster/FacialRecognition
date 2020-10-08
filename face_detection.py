import cv2
from mtcnn.mtcnn import MTCNN

from data_loading import load_images
import sys
import joblib

from main import test_with_svm


def predict_with_classifier(file):
    classifier = joblib.load("models/classifier.jolib")
    preprocessing = joblib.load("models/preprocessing.jolib")
    pca_processing = joblib.load("models/pca_processing.jolib")
    _, labels, names = load_images()
    images, labels_test, names_test = load_images(file)
    return test_with_svm(images, classifier, preprocessing, pca_processing, False,
                         labels_test, names_test=names_test, names=names)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        filename = "challenges/maru_nachito.jpg"
    else:
        filename = sys.argv[1]

    detector = MTCNN()
    # img = cv2.imread('examples/benedict_test.jpg')
    img = cv2.imread(filename)
    faces = detector.detect_faces(img)  # resulttw

    counter = 0
    # Draw faces on image
    if len(faces) == 0:
        print("No faces detected")
        exit(1)

    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        path_to_file = f"detection/person{counter}.jpg"
        cv2.imwrite(path_to_file, img[y:y1, x:x1])
        # resizeImage(path_to_file)
        counter += 1

    labels = predict_with_classifier("detection")
    i = 0
    while i != counter:
        image = cv2.imread(f"detection/person{i}.jpg")
        cv2.imwrite(f"detection/{labels[i]}{i}.jpg", image)
        i += 1

    cv2.imwrite("complete_image.jpg", img)
