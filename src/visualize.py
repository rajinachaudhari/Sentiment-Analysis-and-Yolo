import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(metrics_dict, save_path):
    df = pd.DataFrame(metrics_dict.items(), columns=["Metric", "Value"])

    plt.figure(figsize=(10,6))
    plt.barh(df["Metric"], df["Value"])
    plt.title("License Plate Detection Metrics")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
