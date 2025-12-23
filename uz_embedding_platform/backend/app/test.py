from app.models.model_manager import load_model

model = load_model()

print(model.get_word_vector("salom"))
