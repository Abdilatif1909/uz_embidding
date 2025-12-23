import fasttext

class FastTextVectorizer:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def get_vector(self, text):
        return self.model.get_sentence_vector(text)
def batch_vectorize_bert(model, texts):
    vectors = []
    for t in texts:
        vectors.append(model.get_sentence_vector(t))
    return vectors
