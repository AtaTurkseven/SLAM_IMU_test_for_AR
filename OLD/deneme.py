import pandas as pd
import numpy as np

df = pd.read_csv("imu_only.csv")

t = df["imu_time_us"].to_numpy() * 1e-6
t = t - t[0]
dt = np.diff(t)

print("Duration:", t[-1])
print("Rows:", len(df))
print("Median rate:", 1.0 / np.median(dt))
print("Mean rate:", len(df) / t[-1])
print("dt min:", dt.min())
print("dt max:", dt.max())
print("dt std:", dt.std())

bad = np.where(dt > 0.05)[0]
print("Gaps over 50ms:", len(bad))

for i in bad[:20]:
    print("gap at row", i, "dt =", dt[i])