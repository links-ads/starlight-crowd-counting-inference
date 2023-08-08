# Crowd Counting Inference

![Input and output example of crowded inference](./data/poster.png)
This repository contains code for performing crowd counting inference using a pre-trained model. The code allows you to input an image and obtain estimated head localization points in crowded scenes. The inference is based on a trained ONNX model that has been optimized for crowd counting tasks.

## How to Run

1. Clone the repository to your local machine:

```sh
git clone https://github.com/your-username/crowd-counting-inference.git
cd crowd-counting-inference
```

2. Install the required dependencies. It's recommended to use a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Import the library and run the inference:

```python
from crowd_counting_inference import CrowdCountingModel
# ... other imports

# Specify the path to your YAML file
CONFIG_FILE = "path/to/config"

if __name__ == "__main__":
    # Load config
    with open(CONFIG_FILE, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Load the sample image
    img = ...

    # Create and run the model
    runtime = CrowdCountingModel(
        model_path=config["model_path"],
        providers=config["providers"],
        head_threshold=config["head_threshold"],
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
    )
    pred_points = runtime.inference(np.array(img))
```


Every `ONNX` file is associated to a model configuration (a `.yaml` file). Model weights have not been uploaded to the repository due to the large size, if you wish to run the code please reach to `fabio.montello[at]linksfoundation.com`.