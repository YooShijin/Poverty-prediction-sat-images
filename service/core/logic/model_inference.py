import service.main as s
import pickle
import pandas as pd
import cv2

def determine_poverty(image):
    model = pickle.load(open('poverty_prediction.hdf5', 'rb'))
    test_img = cv2.resize(image, (256, 256))
    test_input = test_img.reshape((1, 256, 256, 3))
    output = model.predict(test_input)[0][2]*5
    return {"poverty_index": output}

