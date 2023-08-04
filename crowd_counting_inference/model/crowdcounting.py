import numpy as np
from crowd_counting_inference.model.base import Model
from crowd_counting_inference.utils.runtime import ONNXRuntime
from crowd_counting_inference.utils.structures import HeadLocalizationResult, Point
from typing import List, Optional
from PIL import Image


class CrowdCountingModel(Model):
    """Crowd Counting Model for ONNX Inference

    This model class performs ONNX inference of a given image to estimate head
    counts in crowded scenes.

    Args:
        model_path (str): Path to the ONNX model file.
        providers (Optional[List[str]]): List of execution providers for ONNX Runtime.
            Defaults to ["TensorrtExecutionProvider", "CPUExecutionProvider"].
        head_threshold (float): Threshold to filter out low-confidence head predictions.
            Defaults to 0.5.
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = ["CPUExecutionProvider"],
        head_threshold: float = 0.5,
    ) -> None:
        super(CrowdCountingModel, self).__init__()

        self.batch_size = 1
        self.input_width, self.input_height = 512, 512
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self._build_model(model_path, providers)
        self.head_threshold = head_threshold

    def _preprocessing(self, image: Image) -> np.array:
        """Preprocess the input image for inference.

        Args:
            image (Image): The input image.

        Returns:
            np.array: Preprocessed image as a NumPy array.
        """
        # Preprocess the image
        # Inference only requires to resize and normalize the image
        image = image.crop((0, 0, self.input_width, self.input_height))
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        return image

    def _build_model(
        self,
        model_path: str,
        providers: Optional[List[str]] = ["CPUExecutionProvider"],
    ):
        self.runtime = ONNXRuntime(model_path, providers)

    def _postprocessing(self, outputs: List[np.array]) -> HeadLocalizationResult:
        """Postprocess the ONNX Runtime outputs to obtain head localization results.

        Args:
            outputs (List[np.array]): Output predictions from ONNX Runtime.

        Returns:
            HeadLocalizationResult: Object containing head localization results.
        """
        pred_logits = outputs[0]
        pred_points = outputs[1][0]
        outputs_scores = np.exp(pred_logits) / np.exp(pred_logits).sum(
            axis=-1, keepdims=True
        )
        outputs_scores = outputs_scores[:, :, 1]

        pred_points = pred_points[outputs_scores[0] > self.head_threshold]
        outputs_scores = outputs_scores[0][outputs_scores[0] > self.head_threshold]

        out_list = HeadLocalizationResult()
        for score, points in zip(outputs_scores, pred_points):
            out_list.add(Point(x=points[0], y=points[1], score=score, label="head"))
        return out_list

    def inference(self, image: Image) -> HeadLocalizationResult:
        """Perform inference on the input image.

        Args:
            image (Image): The input image.

        Returns:
            HeadLocalizationResult: Object containing head localization results.
        """
        image = self._preprocessing(image)
        out = self.runtime.run(image)
        return self._postprocessing(out)
