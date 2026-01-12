from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_metrics
from src.inference import run_inference
from src.config import RESULTS_DIR

def main():
    # train_model()

    metrics = evaluate_model()
    plot_metrics(metrics, RESULTS_DIR / "plots/metrics.png")

    run_inference("data/test/images/Cars42.png")

if __name__ == "__main__":
    main()
