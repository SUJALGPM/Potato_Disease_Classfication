from src.train import train_model
from src.evaluate import evaluate_model
from src.data_preprocessing import get_data_generators

if __name__ == "__main__":
    model, _ = train_model()
    _, val_gen = get_data_generators("dataset")
    evaluate_model(model, val_gen)
