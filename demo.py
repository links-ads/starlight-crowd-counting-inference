from crowd_counting_inference import CrowdCountingModel
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Load the sample image
    sample_image_path = "data/crowd2.jpg"
    img = Image.open(sample_image_path)
    runtime = CrowdCountingModel("weights/crowdcounting_v1.onnx")
    pred_points = runtime.inference(img)

    img_to_draw = np.array(img).copy()

    # draw the predictions
    for p in pred_points.points:
        img_to_draw = cv2.circle(img_to_draw, (int(p.x), int(p.y)), 4, (255, 0, 0), -1)

    # save the visualized image
    plt.imsave("./data/output_test.jpg", img_to_draw)
