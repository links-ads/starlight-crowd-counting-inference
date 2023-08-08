from crowd_counting_inference import CrowdCountingModel
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Specify the path to your YAML file
CONFIG_FILE = "./configs/crowdcounting_v1_batch_1_size_512.yaml"

if __name__ == "__main__":
    # Load config
    with open(CONFIG_FILE, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Load the sample image
    sample_image_path = "data/crowd3.jpg"
    img = Image.open(sample_image_path)

    # Create and run the model
    runtime = CrowdCountingModel(
        model_path=config["model_path"],
        providers=config["providers"],
        head_threshold=config["head_threshold"],
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
    )
    pred_points = runtime.inference(np.array(img))

    # draw the predictions
    img_to_draw = np.array(img).copy()
    for p in pred_points.points:
        img_to_draw = cv2.circle(img_to_draw, (int(p.x), int(p.y)), 4, (255, 0, 0), -1)

    # save the visualized image
    plt.imsave(sample_image_path.replace(".jpg", "_output.jpg"), img_to_draw)
