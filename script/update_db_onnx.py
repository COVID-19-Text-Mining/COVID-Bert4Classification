import argparse
import logging
import os
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
from typing import Callable

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from torch.utils.data import DataLoader
from transformers import AutoConfig, RobertaTokenizerFast

from modeling_multi_label.config import PRETRAINED_MODEL
from modeling_multi_label.dataset import MultiLabelDataCollator
from modeling_multi_label.utils import root_dir, timer, nop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(message)s")
logger = logging.getLogger(__name__)

spec = spec_from_loader(
    "update_utils",
    SourceFileLoader("update_utils", root_dir("script", "update_db.py"))
)
update_utils = module_from_spec(spec)
spec.loader.exec_module(update_utils)

UpdateArgumentParser: argparse.ArgumentParser = update_utils.UpdateArgumentParser
get_collections: Callable = update_utils.get_collections
compute_model_hash: Callable = update_utils.compute_model_hash
get_iterable_dataset: Callable = update_utils.get_iterable_dataset
write_to_db: Callable = update_utils.write_to_db


class UpdateONNXArgumentParser(UpdateArgumentParser):
    def __init__(self, *args, **kwargs):
        super(UpdateONNXArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument("--gpu", action="store_true",
                          help="Whether to enable GPU for inference.")


def create_model_for_provider(model_path: str, provider: str = "CPUExecutionProvider") -> InferenceSession:
    assert provider in get_all_providers(), "provider {} not found, {}".format(provider, get_all_providers())

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


if __name__ == '__main__':
    cli_args = UpdateONNXArgumentParser(
        prog="Update DB ONNX Version", default_path=root_dir("bst_model_onnx")
    ).parse_args()
    cli_args.model_path = os.path.join(cli_args.model_dir, "model-optimized-quantized.onnx")

    _model = create_model_for_provider(
        model_path=cli_args.model_path,
        provider="CPUExecutionProvider" if not cli_args.gpu else "CUDAExecutionProvider"
    )
    model_hash_ = compute_model_hash(cli_args.model_path)

    collection_, output_collection_ = get_collections(
        collection_name=cli_args.collection,
        output_collection_name=cli_args.output_collection,
        debug=cli_args.debug,
    )
    dataset = get_iterable_dataset(
        collection=collection_,
        output_collection=output_collection_,
        debug=cli_args.debug,
    )
    data_collator = MultiLabelDataCollator(
        tokenizer=RobertaTokenizerFast.from_pretrained(cli_args.model_dir),
        return_tensors="np",
    )
    data_loader = DataLoader(dataset=dataset, batch_size=cli_args.batch_size, collate_fn=data_collator)

    model_config = AutoConfig.from_pretrained(cli_args.model_dir)

    with timer("DEBUG:") if cli_args.debug else nop():
        for batch in data_loader:
            ids = batch.pop("_ids")
            logits = _model.run(None, batch)[0]
            write_to_db(
                ids=ids,
                logits=logits,
                id2label=model_config.id2label,
                model_hash=model_hash_,
                collection=collection_,
                output_collection=output_collection_,
                debug=cli_args.debug,
            )
