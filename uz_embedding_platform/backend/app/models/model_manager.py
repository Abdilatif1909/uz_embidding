import os
import fasttext

from app.models.bert_wrapper import BERTEmbedding

loaded_models = {}

def load_model(model_type="fasttext"):

    if model_type in loaded_models:
        return loaded_models[model_type]

    if model_type == "fasttext-en":
        model_path = os.path.join(
            os.path.dirname(__file__),
            "../../embeddings/cc.en.300.bin/cc.en.300.bin"
        )
        print("Loading English FastText...")
        model = fasttext.load_model(model_path)

    elif model_type == "fasttext-uz":
        model_path = os.path.join(
            os.path.dirname(__file__),
            "../../embeddings/cc.uz.300.bin"
        )
        print("Loading Uzbek FastText...")
        model = fasttext.load_model(model_path)

    elif model_type == "bert":
        print("Loading mBERT...")
        model = BERTEmbedding()

    else:
        raise ValueError("Unknown model type")

    loaded_models[model_type] = model
    return model
