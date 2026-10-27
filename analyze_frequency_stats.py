import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

# Load previously calculated FFT results
df = pd.read_csv("frequency_comparison.csv")

fake = df.loc[
    df["class"] == "FAKE", "high_frequency_ratio"
].dropna()

real = df.loc[
    df["class"] == "REAL", "high_frequency_ratio"
].dropna()

if len(fake) < 2 or len(real) < 2:
    raise ValueError("Not enough observations in both classes.")

# Welch's t-test compares class means
t_stat, t_pvalue = ttest_ind(
    fake, real, equal_var=False
)

# Mann-Whitney U tests for a distributional difference
u_stat, u_pvalue = mannwhitneyu(
    fake, real, alternative="two-sided"
)

# Pooled standard deviation and standardized mean difference
pooled_std = (
    (
        (len(fake) - 1) * fake.var(ddof=1)
        + (len(real) - 1) * real.var(ddof=1)
    )
    / (len(fake) + len(real) - 2)
) ** 0.5

cohens_d = (
    (fake.mean() - real.mean()) / pooled_std
    if pooled_std > 0 else float("nan")
)

print("========== FFT FEATURE STATISTICS ==========")
print(f"FAKE images: {len(fake)}")
print(f"REAL images: {len(real)}")
print(f"FAKE mean: {fake.mean():.6f}")
print(f"REAL mean: {real.mean():.6f}")
print(f"Mean difference (FAKE - REAL): {fake.mean() - real.mean():.6f}")
print(f"Welch t-test p-value: {t_pvalue:.6f}")
print(f"Mann-Whitney p-value: {u_pvalue:.6f}")
print(f"Cohen's d: {cohens_d:.6f}")

print("\nInterpretation:")
print("These tests describe the current sample only.")
print("Frames from the same videos are related, so these")
print("p-values do not account for video-level dependence.")
print("A small p-value would not prove useful detection.")
