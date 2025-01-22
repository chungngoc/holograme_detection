import matplotlib.pyplot as plt

def parse_results(file_path):
    """
    Parses the results from the file and computes summary statistics.

    Parameters:
        file_path: Path to the text file containing results.

    Returns:
        data: List of tuples [(folder, origins_count, fraud_count)].
        mean_accuracy: The mean accuracy reported in the file.
    """
    data = []
    mean_accuracy = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Mean accuracy"):
                # Extract mean accuracy
                mean_accuracy = float(line.split(":")[-1].strip())
            else:
                # Parse folder, origins_count, and fraud_count
                folder, origins_count, fraud_count = line.split(", ")
                data.append((folder, int(origins_count), int(fraud_count)))

    return data, mean_accuracy


def summarize_results(data):
    """
    Summarizes the parsed results.

    Parameters:
        data: List of tuples [(folder, origins_count, fraud_count)].

    Returns:
        summary: Dictionary containing total predictions and percentages.
    """
    total_origins = sum(row[1] for row in data)
    total_fraud = sum(row[2] for row in data)
    total_predictions = total_origins + total_fraud

    summary = {
        "total_origins": total_origins,
        "total_fraud": total_fraud,
        "total_predictions": total_predictions,
        "percentage_origins": total_origins / total_predictions * 100,
        "percentage_fraud": total_fraud / total_predictions * 100,
    }

    return summary


def plot_results(data, summary, output_dir="visualization"):
    """
    Plots the results as bar charts and saves them.

    Parameters:
        data: List of tuples [(folder, origins_count, fraud_count)].
        summary: Dictionary containing summary statistics.
        output_dir: Directory to save the visualizations.

    Returns:
        None
    """
    # Create bar chart for folder-wise predictions
    folders = [row[0] for row in data]
    origins_counts = [row[1] for row in data]
    fraud_counts = [row[2] for row in data]

    x = range(len(folders))

    plt.figure(figsize=(12, 6))
    plt.bar(x, origins_counts, label="Origins", color="green", alpha=0.7)
    plt.bar(x, fraud_counts, label="Fraud", color="red", alpha=0.7, bottom=origins_counts)
    plt.xticks(x, folders, rotation=90)
    plt.ylabel("Count")
    plt.title("Predictions per Folder")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions_per_folder.png")
    plt.show()

    # Create pie chart for overall distribution
    plt.figure(figsize=(6, 6))
    plt.pie(
        [summary["total_origins"], summary["total_fraud"]],
        labels=["Origins", "Fraud"],
        colors=["green", "red"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Overall Prediction Distribution")
    plt.savefig(f"{output_dir}/overall_distribution.png")
    plt.show()


if __name__ == '__main__':
    # path to file text contains predictions of all videos, got after running evaluate/evaluate.py
    file_path = "results/evaluate_evaluate_origins.txt"
    data, mean_accuracy = parse_results(file_path)

    # Summarize the data
    summary = summarize_results(data)
    print("Summary:")
    print(summary)
    print(f"Mean Accuracy: {mean_accuracy:.4f}")

    # Plot and save results
    plot_results(data, summary)
