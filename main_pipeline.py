from preprocess import load_and_save_data
from train import train_models
from model_selection import select_and_save_best_model

def run_pipeline():
    load_and_save_data()
    results = train_models()
    select_and_save_best_model(results)
    print("\nPipeline execution completed.")

if __name__ == "__main__":
    run_pipeline()