import pandas
import numpy as np

def ece_equal_width(df: pandas.DataFrame, num_bins=10) -> float:

    # Create bins (0-100 split into 10 equal-width bins)
    bins = np.linspace(0, 1, num_bins + 1)

    # Assign each row to a confidence bin
    df['confidence_bin'] = pandas.cut(df['confidence'], bins=bins, labels=False, right=False)

    # Group by bins and calculate mean accuracy and mean confidence
    bin_stats = df.groupby('confidence_bin').agg(
        bin_accuracy=('accuracy', 'mean'),
        bin_confidence=('confidence', 'mean'),
        bin_size=('confidence', 'size')
    ).reset_index()
    bin_stats["bin_weighted_ece"] = abs(bin_stats["bin_accuracy"] - bin_stats["bin_confidence"]) * bin_stats["bin_size"] / df.shape[0]
    ece = sum(bin_stats["bin_weighted_ece"])
    agg_row = pandas.DataFrame({"confidence_bin": [""], "bin_accuracy": [""], "bin_confidence": [""], "bin_size": [df.shape[0]], "bin_weighted_ece": [ece]})
    bin_stats = pandas.concat([bin_stats, agg_row], ignore_index=True)
    bin_stats.to_csv('tmp/simpleqa_equal_width_ece.csv', index=False)
    print(f"Expected Calibration Error (ECE) (Equal-width): {ece:.4f}")
    return bin_stats


def ece_equal_weight(df: pandas.DataFrame, num_bins=10):
    if num_bins > len(df):
        num_bins = int(len(df) / 2)

    # Sort the DataFrame by confidence
    df_sorted = df.sort_values(by='confidence')

    bins = np.array_split(df_sorted, num_bins)
    
    # Assign each row to a bin (based on index position in the split list)
    df_sorted['confidence_bin'] = np.concatenate([np.repeat(i, len(bin)) for i, bin in enumerate(bins)])
    
    # Group by bins and calculate mean accuracy, mean confidence, and bin size
    bin_stats = df_sorted.groupby('confidence_bin').agg(
        bin_accuracy=('accuracy', 'mean'),
        bin_confidence=('confidence', 'mean'),
        bin_size=('confidence', 'size')  # Count the number of samples in each bin
    ).reset_index()
    bin_stats["bin_weighted_ece"] = abs(bin_stats["bin_accuracy"] - bin_stats["bin_confidence"]) * bin_stats["bin_size"] / df.shape[0]
    ece = sum(bin_stats["bin_weighted_ece"])
    agg_row = pandas.DataFrame({"confidence_bin": [""], "bin_accuracy": [""], "bin_confidence": [""], "bin_size": [df.shape[0]], "bin_weighted_ece": [ece]})
    bin_stats = pandas.concat([bin_stats, agg_row], ignore_index=True)
    bin_stats.to_csv('tmp/simpleqa_equal_weight_ece.csv', index=False)
    print(f"Expected Calibration Error (ECE) (Equal-weighted): {ece:.4f}")
    return bin_stats