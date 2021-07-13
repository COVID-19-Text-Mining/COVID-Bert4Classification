# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
from argparse import ArgumentParser
from os import listdir, makedirs, remove
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from packaging.version import Version, parse
from transformers.file_utils import ModelOutput, is_tf_available, is_torch_available
from transformers.pipelines import Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding

# This is the minimal required version to
# support some ONNX Runtime features
from modeling_multi_label.model import MultiLabelModel

ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")


SUPPORTED_PIPELINES = [
    "feature-extraction",
    "ner",
    "sentiment-analysis",
    "fill-mask",
    "question-answering",
    "text-generation",
    "translation_en_to_fr",
    "translation_en_to_de",
    "translation_en_to_ro",
]


class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """

    def __init__(self):
        super().__init__("ONNX Converter")

        self.add_argument(
            "--pipeline",
            type=str,
            choices=SUPPORTED_PIPELINES,
            default="feature-extraction",
        )
        self.add_argument(
            "--model",
            type=str,
            required=True,
            help="Model's id or path (ex: bert-base-cased)",
        )
        self.add_argument("--tokenizer", type=str, help="Tokenizer's id or path (ex: bert-base-cased)")
        self.add_argument(
            "--framework",
            type=str,
            choices=["pt", "tf"],
            help="Framework for loading the model",
        )
        self.add_argument("--opset", type=int, default=11, help="ONNX opset to use")
        self.add_argument(
            "--check-loading",
            action="store_true",
            help="Check ONNX is able to load the model",
        )
        self.add_argument(
            "--use-external-format",
            action="store_true",
            help="Allow exporting model >= than 2Gb",
        )
        self.add_argument(
            "--quantize",
            action="store_true",
            help="Quantize the neural network to be run with int8",
        )

        self.add_argument(
            "--remove-unoptimized",
            action="store_true",
            help="only stores the last optimized model"
        )
        self.add_argument("output")


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Append a string-identifier at the end (before the extension, if any) to the provided filepath

    Args:
        filename: pathlib.Path The actual path object we would like to add an identifier suffix
        identifier: The suffix to add

    Returns: String with concatenated identifier at the end of the filename
    """
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        import onnxruntime

        # Parse the version of the installed onnxruntime
        ort_version = parse(onnxruntime.__version__)

        # We require 1.4.0 minimum
        if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
            raise ImportError(
                "We found an older version of onnxruntime ({}) "
                "but we require onnxruntime to be >= {} to enable all the conversions options.\n"
                "Please update onnxruntime by running `pip install --upgrade onnxruntime`".format(
                    onnxruntime.__version__, minimum_version
                )
            )

    except ImportError:
        raise ImportError(
            "onnxruntime doesn't seem to be currently installed. "
            "Please install the onnxruntime by running `pip install onnxruntime`"
            " and relaunch the conversion."
        )


def ensure_valid_input(model, tokens, input_names):
    """
    Ensure input are presented in the correct order, without any Non

    Args:
        model: The model used to forward the input data
        tokens: BatchEncoding holding the input data
        input_names: The name of the inputs

    Returns: Tuple

    """
    import inspect
    print("Ensuring inputs are in correct order")

    model_args_name = tuple(inspect.signature(model.forward).parameters.keys())

    model_args, ordered_input_names = [], []
    for arg_name in model_args_name:
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            print("{} is not present in the generated input list.".format(arg_name))

    print("Generated inputs order: {}".format(ordered_input_names))
    return ordered_input_names, tuple(model_args)


def infer_shapes(nlp: Pipeline, framework: str) -> Tuple[List[str], List[str], Dict, BatchEncoding]:
    """
    Attempt to infer the static vs dynamic axes for each input and output tensors for a specific model

    Args:
        nlp: The pipeline object holding the model to be exported
        framework: The framework identifier to dispatch to the correct inference scheme (pt/tf)

    Returns:

        - List of the inferred input variable names
        - List of the inferred output variable names
        - Dictionary with input/output variables names as key and shape tensor as value
        - a BatchEncoding reference which was used to infer all the above information
    """

    def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
        if isinstance(tensor, (tuple, list)):
            return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]

        else:
            # Let's assume batch is the first axis with only 1 element (~~ might not be always true ...)
            axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == 1][0]: "batch"}
            if is_input:
                if len(tensor.shape) == 2:
                    axes[1] = "sequence"
                else:
                    raise ValueError("Unable to infer tensor axes ({})".format(len(tensor.shape)))
            else:
                seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
                axes.update({dim: "sequence" for dim in seq_axes})

        print("Found {} {name} with shape: {axes}".format(
            'input' if is_input else 'output',
            name=name,
            axes=axes,
        ))
        return axes

    tokens = nlp.tokenizer(
        "This is a sample output",
        return_tensors=framework,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
    )
    seq_len = tokens.input_ids.shape[-1]
    outputs = nlp.model(**tokens) if framework == "pt" else nlp.model(tokens)
    if isinstance(outputs, ModelOutput):
        outputs = outputs.to_tuple()
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    # Generate input names & axes
    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: build_shape_dict(k, v, True, seq_len) for k, v in tokens.items()}

    # flatten potentially grouped outputs (past for gpt2, attentions)
    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)

    # Generate output names & axes
    output_names = ["output_{}".format(i) for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for k, v in zip(output_names, outputs_flat)}

    # Create the aggregated axes representation
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_vars, output_names, dynamic_axes, tokens


def load_graph_from_args(
    pipeline_name: str, framework: str, model_name: str, tokenizer: Optional[str] = None, **models_kwargs
) -> Pipeline:
    """
    Convert the set of arguments provided through the CLI to an actual pipeline reference (tokenizer + model

    Args:
        pipeline_name: The kind of pipeline to use (ner, question-answering, etc.)
        framework: The actual model to convert the pipeline from ("pt" or "tf")
        model_name: The model name which will be loaded by the pipeline
        tokenizer: The tokenizer name which will be loaded by the pipeline, default to the model's value

    Returns: Pipeline object

    """
    # If no tokenizer provided
    if tokenizer is None:
        tokenizer = model_name

    model = MultiLabelModel.from_pretrained(
        model_name
    )

    # Check the wanted framework is available
    if framework == "pt" and not is_torch_available():
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")
    if framework == "tf" and not is_tf_available():
        raise Exception("Cannot convert because TF is not installed. Please install tensorflow first.")

    print("Loading pipeline (model: {}, tokenizer: {})".format(model_name, tokenizer))

    # Allocate tokenizer and model
    return pipeline(pipeline_name, model=model, tokenizer=tokenizer, framework=framework, model_kwargs=models_kwargs)


def convert_pytorch(nlp: Pipeline, opset: int, output: Path, use_external_format: bool):
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB

    Returns:

    """
    if not is_torch_available():
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")

    import torch
    from torch.onnx import export

    print("Using framework PyTorch: {}".format(torch.__version__))

    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "pt")
        ordered_input_names, model_args = ensure_valid_input(nlp.model, tokens, input_names)

        export(
            nlp.model,
            model_args,
            f=output.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )


def convert_tensorflow(nlp: Pipeline, opset: int, output: Path):
    """
    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model

    Notes: TensorFlow cannot export model bigger than 2GB due to internal constraint from TensorFlow

    """
    if not is_tf_available():
        raise Exception("Cannot convert because TF is not installed. Please install tensorflow first.")

    print("/!\\ Please note TensorFlow doesn't support exporting model > 2Gb /!\\")

    try:
        import tensorflow as tf

        from keras2onnx import __version__ as k2ov
        from keras2onnx import convert_keras, save_model

        print("Using framework TensorFlow: {}, keras2onnx: {}".format(tf.version.VERSION, k2ov))

        # Build
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "tf")

        # Forward
        nlp.model.predict(tokens.data)
        onnx_model = convert_keras(nlp.model, nlp.model.name, target_opset=opset)
        save_model(onnx_model, output.as_posix())

    except ImportError as e:
        raise Exception(
            "Cannot import {0} required to convert TF model to ONNX. Please install {0} first.".format(e.name)
        )


def convert(
    framework: str,
    model: str,
    output: Path,
    opset: int,
    tokenizer: Optional[str] = None,
    use_external_format: bool = False,
    pipeline_name: str = "feature-extraction",
    **model_kwargs
):
    """
    Convert the pipeline object to the ONNX Intermediate Representation (IR) format

    Args:
        framework: The framework the pipeline is backed by ("pt" or "tf")
        model: The name of the model to load for the pipeline
        output: The path where the ONNX graph will be stored
        opset: The actual version of the ONNX operator set to use
        tokenizer: The name of the model to load for the pipeline, default to the model's name if not provided
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB (PyTorch only)
        pipeline_name: The kind of pipeline to instantiate (ner, question-answering, etc.)
        model_kwargs: Keyword arguments to be forwarded to the model constructor

    Returns:

    """
    print("ONNX opset version set to: {}".format(opset))

    # Load the pipeline
    nlp = load_graph_from_args(pipeline_name, framework, model, tokenizer, **model_kwargs)

    if not output.parent.exists():
        print("Creating folder {}".format(output.parent))
        makedirs(output.parent.as_posix())
    elif len(listdir(output.parent.as_posix())) > 0:
        raise Exception("Folder {} is not empty, aborting conversion".format(output.parent.as_posix()))

    # Export the graph
    if framework == "pt":
        convert_pytorch(nlp, opset, output, use_external_format)
    else:
        convert_tensorflow(nlp, opset, output)


def optimize(onnx_model_path: Path) -> Path:
    """
    Load the model at the specified path and let onnxruntime look at transformations on the graph to enable all the
    optimizations possible

    Args:
        onnx_model_path: filepath where the model binary description is stored

    Returns: Path where the optimized model binary description has been saved

    """

    from onnxruntime.transformers.optimizer import optimize_model
    from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions

    # various graph optimization options.
    # ZCode uses all the optimizations by default.
    # Whether to use quantization or not can be selected optionally.
    optimization_options = BertOptimizationOptions('bert')
    optimization_options.enable_gelu = True
    optimization_options.enable_layer_norm = True
    optimization_options.enable_attention = True
    optimization_options.enable_skip_layer_norm = True
    optimization_options.enable_embed_layer_norm = True
    optimization_options.enable_bias_skip_layer_norm = True
    optimization_options.enable_bias_gelu = True
    optimization_options.enable_gelu_approximation = False

    optimizer = optimize_model(onnx_model_path.as_posix(),
                               model_type='bert',
                               num_heads=0,
                               hidden_size=0,
                               optimization_options=optimization_options,
                               opt_level=0,
                               use_gpu=False,
                               only_onnxruntime=False)

    opt_model_path = generate_identified_filename(onnx_model_path, "-optimized")
    optimizer.save_model_to_file(opt_model_path.as_posix())

    print("Optimized model has been written at {}: \N{heavy check mark}".format(opt_model_path))
    print("/!\\ Optimized model contains hardware specific operators which might not be portable. /!\\")

    return opt_model_path


def quantize(onnx_model_path: Path) -> Path:
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

    Args:
        onnx_model_path: Path to location the exported ONNX model is stored

    Returns: The Path generated for the quantized
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Append "-quantized" at the end of the model's name
    quantized_model_path = generate_identified_filename(onnx_model_path, "-quantized")

    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     op_types_to_quantize=['MatMul', 'Attention'],
                     weight_type=QuantType.QInt8,
                     per_channel=True,
                     reduce_range=True,
                     nodes_to_exclude=[],
                     extra_options={'WeightSymmetric': False, 'MatMulConstBOnly': True})

    print("Quantized model has been written at {}: \N{heavy check mark}".format(quantized_model_path))
    return quantized_model_path


def verify(path: Path):
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

    print("Checking ONNX model loading from: {} ...".format(path))
    try:
        onnx_options = SessionOptions()
        _ = InferenceSession(path.as_posix(), onnx_options, providers=["CPUExecutionProvider"])
        print("Model {} correctly loaded: \N{heavy check mark}".format(path))
    except RuntimeException as re:
        print("Error while loading the model {}: \N{heavy ballot x}".format(re))


def save_config(model, dst):
    from transformers import AutoConfig, PretrainedConfig

    config: PretrainedConfig = AutoConfig.from_pretrained(model)
    config.save_pretrained(dst)
    print("\n===== Save model config to {} =====".format(dst))


if __name__ == "__main__":
    parser = OnnxConverterArgumentParser()
    args = parser.parse_args()

    # Make sure output is absolute path
    args.output = Path(args.output).absolute()

    print("\n====== Converting model to ONNX ======")
    # Convert
    convert(
        args.framework,
        args.model,
        args.output,
        args.opset,
        args.tokenizer,
        args.use_external_format,
        args.pipeline,
    )

    if args.quantize:
        # Ensure requirements for quantization on onnxruntime is met
        check_onnxruntime_requirements(ORT_QUANTIZE_MINIMUM_VERSION)

        # onnxruntime optimizations doesn't provide the same level of performances on TensorFlow than PyTorch
        if args.framework == "tf":
            print(
                "\t Using TensorFlow might not provide the same optimization level compared to PyTorch.\n"
                "\t For TensorFlow users you can try optimizing the model directly through onnxruntime_tools.\n"
                "\t For more information, please refer to the onnxruntime documentation:\n"
                "\t\thttps://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers\n"
            )

        print("\n====== Optimizing ONNX model ======")

        # Quantization works best when using the optimized version of the model
        args.optimized_output = optimize(args.output)

        # Do the quantization on the right graph
        args.quantized_output = quantize(args.optimized_output)

    # And verify
    if args.check_loading:
        print("\n====== Check exported ONNX model(s) ======")
        verify(args.output)

        if hasattr(args, "optimized_output"):
            verify(args.optimized_output)

        if hasattr(args, "quantized_output"):
            verify(args.quantized_output)

    if args.remove_unoptimized:
        files = listdir(args.output.parent.as_posix())
        files.sort(key=lambda _: len(_))

        for file in files[:-1]:
            remove(path=args.output.parent.joinpath(file).absolute().as_posix())

    save_config(args.model, args.output.parent.as_posix())
