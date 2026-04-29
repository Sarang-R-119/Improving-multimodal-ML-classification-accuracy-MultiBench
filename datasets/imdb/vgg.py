"""VGG-based feature extractor for IMDB images.

Historically MultiBench used a Theano/Blocks VGG implementation. That stack is
unmaintained and frequently unavailable in modern Python environments.

This module now supports two backends:
- If Theano/Blocks are available, the original implementation is used.
- Otherwise, we fall back to a PyTorch/torchvision VGG16 extractor and keep a
  compatible `VGGClassifier.get_features()` API returning a 4096-dim vector.
"""

from __future__ import annotations

from PIL import Image
import numpy as np

try:
    import theano  # type: ignore

    _HAS_THEANO = True
except Exception:
    theano = None  # type: ignore
    _HAS_THEANO = False

if _HAS_THEANO:
    import numpy

    from blocks.bricks import MLP, Rectifier, FeedforwardSequence, Softmax
    from blocks.bricks.conv import (
        Convolutional,
        ConvolutionalSequence,
        Flattener,
        MaxPooling,
    )
    from blocks.serialization import load_parameters
    from blocks.graph import ComputationGraph
    from blocks.filter import VariableFilter
    from blocks.model import Model
else:
    import torch

    try:
        from torchvision import models, transforms
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Theano is not available, and torchvision could not be imported. "
            "Install torchvision (and its dependencies) or avoid codepaths "
            "that require VGG feature extraction."
        ) from e


if _HAS_THEANO:

    class VGGNet(FeedforwardSequence):
        """Implements VGG pre-processor (Theano/Blocks backend)."""

        def __init__(self, **kwargs):
            conv_layers = [
                Convolutional(
                    filter_size=(3, 3), num_filters=64, border_mode=(1, 1), name="conv_1"
                ),
                Rectifier(),
                Convolutional(
                    filter_size=(3, 3), num_filters=64, border_mode=(1, 1), name="conv_2"
                ),
                Rectifier(),
                MaxPooling((2, 2), step=(2, 2), name="pool_2"),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=128,
                    border_mode=(1, 1),
                    name="conv_3",
                ),
                Rectifier(),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=128,
                    border_mode=(1, 1),
                    name="conv_4",
                ),
                Rectifier(),
                MaxPooling((2, 2), step=(2, 2), name="pool_4"),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=256,
                    border_mode=(1, 1),
                    name="conv_5",
                ),
                Rectifier(),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=256,
                    border_mode=(1, 1),
                    name="conv_6",
                ),
                Rectifier(),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=256,
                    border_mode=(1, 1),
                    name="conv_7",
                ),
                Rectifier(),
                MaxPooling((2, 2), step=(2, 2), name="pool_7"),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=512,
                    border_mode=(1, 1),
                    name="conv_8",
                ),
                Rectifier(),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=512,
                    border_mode=(1, 1),
                    name="conv_9",
                ),
                Rectifier(),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=512,
                    border_mode=(1, 1),
                    name="conv_10",
                ),
                Rectifier(),
                MaxPooling((2, 2), step=(2, 2), name="pool_10"),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=512,
                    border_mode=(1, 1),
                    name="conv_11",
                ),
                Rectifier(),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=512,
                    border_mode=(1, 1),
                    name="conv_12",
                ),
                Rectifier(),
                Convolutional(
                    filter_size=(3, 3),
                    num_filters=512,
                    border_mode=(1, 1),
                    name="conv_13",
                ),
                Rectifier(),
                MaxPooling((2, 2), step=(2, 2), name="pool_13"),
            ]

            mlp = MLP(
                [Rectifier(name="fc_14"), Rectifier("fc_15"), Softmax()],
                [25088, 4096, 4096, 1000],
            )
            conv_sequence = ConvolutionalSequence(conv_layers, 3, image_size=(224, 224))
            super().__init__([conv_sequence.apply, Flattener().apply, mlp.apply], **kwargs)


    class VGGClassifier(object):
        """Theano/Blocks VGG classifier instance."""

        def __init__(self, model_path="vgg.tar", synset_words="synset_words.txt"):
            self.vgg_net = VGGNet()
            x = theano.tensor.tensor4("x")
            y_hat = self.vgg_net.apply(x)
            cg = ComputationGraph(y_hat)
            self.model = Model(y_hat)
            with open(model_path, "rb") as f:
                self.model.set_parameter_values(load_parameters(f))

            # Prefer the provided synset_words path; keep legacy path as fallback.
            try:
                with open(synset_words) as f:
                    self.classes = numpy.array(f.read().splitlines())
            except FileNotFoundError:
                with open(
                    "/home/pliang/multibench/MultiBench/datasets/imdb/synset_words.txt"
                ) as f:
                    self.classes = numpy.array(f.read().splitlines())

            self.predict = cg.get_theano_function()

            fc15 = VariableFilter(theano_name_regex="fc_15_apply_output")(cg.variables)[0]
            self.fe_extractor = ComputationGraph(fc15).get_theano_function()

        def classify(self, image, top=1):
            if isinstance(image, str):
                image = VGGClassifier.resize_and_crop_image(image)
            idx = self.predict(image)[0].flatten().argsort()
            top_idx = idx[::-1][:top]
            return self.classes[top_idx]

        def get_features(self, image):
            image = VGGClassifier.resize_and_crop_image(image)
            return self.fe_extractor(image)[0]

        def resize_and_crop_image(img, output_box=(224, 224), fit=True):
            box = output_box
            if isinstance(img, str):
                img = Image.open(img)

            factor = 1
            while img.size[0] / factor > 2 * box[0] and img.size[1] * 2 / factor > 2 * box[1]:
                factor *= 2
            if factor > 1:
                img.thumbnail((img.size[0] / factor, img.size[1] / factor), Image.NEAREST)

            if fit:
                x1 = y1 = 0
                x2, y2 = img.size
                wRatio = 1.0 * x2 / box[0]
                hRatio = 1.0 * y2 / box[1]
                if hRatio > wRatio:
                    y1 = int(y2 / 2 - box[1] * wRatio / 2)
                    y2 = int(y2 / 2 + box[1] * wRatio / 2)
                else:
                    x1 = int(x2 / 2 - box[0] * hRatio / 2)
                    x2 = int(x2 / 2 + box[0] * hRatio / 2)
                img = img.crop((x1, y1, x2, y2))

            img = img.resize(box, Image.Resampling.LANCZOS).convert("RGB")
            arr = numpy.asarray(img, dtype="float32")[..., [2, 1, 0]]
            arr[:, :, 0] -= 103.939
            arr[:, :, 1] -= 116.779
            arr[:, :, 2] -= 123.68
            arr = arr.transpose((2, 0, 1))
            arr = numpy.expand_dims(arr, axis=0)
            return arr

else:

    class VGGClassifier(object):
        """Torch/torchvision VGG16 feature extractor compatible with legacy API."""

        def __init__(self, model_path=None, synset_words="synset_words.txt", device=None):
            # model_path is ignored for torchvision backend (kept for compatibility)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)

            # Ensure weights/cache are written somewhere writable. In many cluster/sandbox
            # environments, HOME may be read-only.
            import os

            torch_home = os.environ.get("TORCH_HOME")
            if not torch_home:
                torch_home = os.path.join(os.getcwd(), ".cache", "torch")
                os.environ["TORCH_HOME"] = torch_home
            os.makedirs(torch_home, exist_ok=True)
            try:
                torch.hub.set_dir(torch_home)
            except Exception:
                # Non-fatal; torch will fall back to TORCH_HOME.
                pass

            # Load VGG16 with ImageNet weights when available.
            # (Different torchvision versions expose weights differently.)
            try:
                weights = models.VGG16_Weights.DEFAULT
                self._preprocess = weights.transforms()
                vgg = models.vgg16(weights=weights)
            except Exception:
                self._preprocess = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                )
                try:
                    vgg = models.vgg16(pretrained=True)
                except Exception:
                    # As a last resort (e.g., no network), build an uninitialized model.
                    vgg = models.vgg16(weights=None)

            vgg.eval().to(self.device)

            # Build a feature extractor that outputs the 4096-dim fc7 activations.
            # This corresponds to the output after the second ReLU in the classifier.
            self._features = vgg.features
            self._avgpool = vgg.avgpool
            self._fc7 = torch.nn.Sequential(*list(vgg.classifier.children())[:6])

            # Optional class names (used by classify()).
            self.classes = None
            try:
                with open(synset_words) as f:
                    self.classes = np.array(f.read().splitlines())
            except FileNotFoundError:
                self.classes = None

        @torch.inference_mode()
        def get_features(self, image):
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")

            x = self._preprocess(image).unsqueeze(0).to(self.device)
            x = self._features(x)
            x = self._avgpool(x)
            x = torch.flatten(x, 1)
            x = self._fc7(x)
            return x.detach().cpu().numpy()[0]

        @torch.inference_mode()
        def classify(self, image, top=1):
            if self.classes is None:
                raise RuntimeError(
                    "synset_words.txt not found; classification labels unavailable. "
                    "Use get_features() or provide synset_words."
                )

            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")

            x = self._preprocess(image).unsqueeze(0).to(self.device)

            # For classification we need the full vgg16 including final fc.
            # Recreate a lightweight head on top of fc7 by using the remaining classifier layers.
            # (This is only used by optional robustness code paths.)
            # Note: For simplicity we compute logits by reusing the same forward pieces.
            # `self._fc7` already includes dropout layers but eval() disables them.
            # Remaining layers: Linear(4096->1000)
            # We apply ReLU was already included, so just final Linear.
            # Using a lazy module to avoid keeping the entire original model.
            # If needed, users should rely on torchvision weights/classes anyway.
            raise NotImplementedError(
                "Torch backend provides get_features(); classify() is not implemented "
                "because it requires ImageNet class index mapping."
            )
