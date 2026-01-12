# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# def plot_metrics(metrics_dict, save_path):
#     df = pd.DataFrame(metrics_dict.items(), columns=["Metric", "Value"])


#     plt.figure(figsize=(10,6))
#     plt.barh(df["Metric"], df["Value"])
#     plt.title("License Plate Detection Metrics")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(metrics, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ Convert metrics dict to DataFrame
    df = pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    )

    plt.figure(figsize=(10, 6))
    plt.barh(df["Metric"], df["Value"])
    plt.title("License Plate Detection Metrics")
    plt.xlabel("Value")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()   # IMPORTANT: avoids UI issues
