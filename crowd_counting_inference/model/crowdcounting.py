import numpy as np
from crowd_counting_inference.model.base import Model
from crowd_counting_inference.utils.runtime import ONNXRuntime
from crowd_counting_inference.utils.structures import HeadLocalizationResult, Point
from crowd_counting_inference.utils import softmax, ceiling_division
from typing import List, Optional
from PIL import Image


class CrowdCountingModel(Model):
    """Crowd Counting Model for ONNX Inference

    This model class performs ONNX inference on a given image to estimate head
    counts in crowded scenes.

    Args:
        model_path (str): Path to the ONNX model file.
        providers (Optional[List[str]]): List of execution providers for ONNX Runtime.
            Defaults to ["TensorrtExecutionProvider", "CPUExecutionProvider"].
        head_threshold (float): Threshold to filter out low-confidence head predictions.
            Defaults to 0.5.
        patch_size (int): Size of image patches for preprocessing. Defaults to 256.
        batch_size (int): Batch size for inference. Defaults to 8.
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = ["CPUExecutionProvider"],
        head_threshold: float = 0.5,
        patch_size: int = 256,
        batch_size: int = 8,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        super(CrowdCountingModel, self).__init__()

        assert isinstance(model_path, str), "model_path must be a string"
        assert isinstance(providers, list), "providers must be a list"
        assert all(
            isinstance(provider, str) for provider in providers
        ), "providers must contain strings"
        assert isinstance(head_threshold, float), "head_threshold must be a float"
        assert isinstance(patch_size, int), "patch_size must be an int"
        assert isinstance(batch_size, int), "batch_size must be an int"
        assert (
            isinstance(mean, list) and len(mean) == 3
        ), "mean must be a list of 3 floats"
        assert all(
            isinstance(value, float) for value in mean
        ), "mean values must be floats"
        assert isinstance(std, list) and len(std) == 3, "std must be a list of 3 floats"
        assert all(
            isinstance(value, float) for value in std
        ), "std values must be floats"

        self.batch_size = batch_size
        self.mean, self.std = mean, std
        self._build_model(model_path, providers)
        self.head_threshold = head_threshold
        self.patch_size = patch_size
        self.batch_size = batch_size

    @staticmethod
    def _extract_patches(image: np.ndarray, kernel_size: tuple) -> np.ndarray:
        """Extract patches from an image.

        Args:
            image (np.ndarray): Input image.
            kernel_size (tuple): Size of each patch.

        Returns:
            np.ndarray: Array of extracted image patches.
        """
        h, w, c = image.shape
        th, tw = kernel_size

        # Reshape the image into tiles of specified size
        tiled_array = image.reshape(h // th, th, w // tw, tw, c)
        tiled_array = tiled_array.swapaxes(1, 2)  # Swap axes for correct tiling
        return tiled_array.reshape(-1, th, tw, c)

    @staticmethod
    def _pad_image_to_multiple(
        image: np.ndarray, kernel_size: int, pad_value: int = 0
    ) -> np.ndarray:
        """Pad an image to ensure its dimensions are multiples of a specified kernel size.

        Args:
            image (np.ndarray): Input image.
            kernel_size (int): Desired kernel size for padding.
            pad_value (int, optional): Value for padding. Defaults to 0.

        Returns:
            np.ndarray: Padded image.
        """
        H, W, _ = image.shape
        K = kernel_size
        pad_h = (K - (H % K)) % K
        pad_w = (K - (W % K)) % K

        # Pad the image to ensure dimensions are multiples of kernel_size
        padded_image = np.pad(
            image, ((0, pad_h), (0, pad_w), (pad_value, pad_value)), mode="constant"
        )

        return padded_image

    @staticmethod
    def _pad_batch(input_tensor: np.ndarray, target_batch_size: int) -> np.ndarray:
        """Pad a batch of input tensors to match the target batch size.

        Args:
            input_tensor (np.ndarray): Input tensor with shape (N, H, W, C).
            target_batch_size (int): Desired batch size.

        Returns:
            np.ndarray: Padded tensor with shape (B, H, W, C), where B is the target batch size.
        """
        N, H, W, C = input_tensor.shape
        B = target_batch_size

        if N >= B:
            return input_tensor

        empty_images = np.zeros((B - N, H, W, C), dtype=input_tensor.dtype)
        padded_tensor = np.concatenate([input_tensor, empty_images], axis=0)

        return padded_tensor

    def _normalize_images(self, images: np.ndarray) -> np.ndarray:
        """Normalize a batch of images for inference.

        Args:
            images (np.ndarray): Input batch of images with shape (N, H, W, C).

        Returns:
            np.ndarray: Normalized batch of images with shape (N, C, H, W).
        """
        images /= 255.0
        images = (images - self.mean) / self.std
        images = np.transpose(images, (0, 3, 1, 2))
        return images

    def _preprocessing(self, image: Image) -> np.array:
        """Preprocess the input image for inference.

        Args:
            image (Image): The input image.

        Returns:
            np.array: Preprocessed image as a NumPy array.
        """

        image = np.array(image).astype(np.float32)
        image = self._pad_image_to_multiple(image, 512, 0)
        patches = self._extract_patches(image, (self.patch_size, self.patch_size))
        patches = self._normalize_images(patches)
        return patches

    def _build_model(
        self,
        model_path: str,
        providers: Optional[List[str]] = ["CPUExecutionProvider"],
    ) -> None:
        """Build the ONNX Runtime model.

        Args:
            model_path (str): Path to the ONNX model file.
            providers (Optional[List[str]]): List of execution providers for ONNX Runtime.
        """
        self.runtime = ONNXRuntime(model_path, providers)

    def _postprocessing(
        self, points: List[np.array], logits: List[np.array], image_shape: tuple
    ) -> HeadLocalizationResult:
        """Postprocess the ONNX Runtime outputs to obtain head localization results.

        Args:
            outputs (List[np.array]): Output predictions from ONNX Runtime.

        Returns:
            HeadLocalizationResult: Object containing head localization results.
        """
        # Softmax of the logits output
        H, W, _ = image_shape
        logits = softmax(logits)[:, :, 1]

        # Reconstruct original patches to compute x,y shift in the original image
        h_patches = ceiling_division(H, self.patch_size)
        w_patches = ceiling_division(W, self.patch_size)
        points = points.reshape(h_patches, w_patches, *points.shape[1:])
        logits = logits.reshape(h_patches, w_patches, *logits.shape[1:])

        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                points[i, j, :, 0] = points[i, j, :, 0] + self.patch_size * j
                points[i, j, :, 1] = points[i, j, :, 1] + self.patch_size * i

        # Filter only points with a digit higher than theshold
        out_points = points[logits > self.head_threshold]
        out_scores = logits[logits > self.head_threshold]

        # Create output according to the HeadLocalizationResult object
        out_list = HeadLocalizationResult()
        for score, points in zip(out_scores, out_points):
            out_list.add(Point(x=points[0], y=points[1], score=score, label="head"))
        return out_list

    def inference(self, image: Image) -> HeadLocalizationResult:
        """Perform inference on the input image.

        Args:
            image (Image): The input image.

        Returns:
            HeadLocalizationResult: Object containing head localization results.
        """
        # Preprocess the input image
        images = self._preprocessing(image)

        # Initialize variables to store predicted logits and points
        pred_logits_batch = pred_points_batch = None

        # Loop over image batches for inference
        for i in range(ceiling_division(images.shape[0], self.batch_size)):
            batch = images[i * self.batch_size : (i + 1) * self.batch_size, :, :]

            # Pad the batch if its size is less than the specified batch size
            if batch.shape[0] < self.batch_size:
                batch = self._pad_batch(batch, self.batch_size)

            # Run inference using the ONNX Runtime model
            pred_logits, pred_points = self.runtime.run(batch)

            # Concatenate the predictions to the accumulated batch
            pred_logits_batch = (
                np.concatenate([pred_logits_batch, pred_logits], axis=0)
                if pred_logits_batch is not None
                else pred_logits
            )
            pred_points_batch = (
                np.concatenate([pred_points_batch, pred_points], axis=0)
                if pred_points_batch is not None
                else pred_points
            )

        # Postprocess the results and return head localization
        return self._postprocessing(pred_points_batch, pred_logits_batch, image.shape)
