#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- input file ---
filename = "postProcessing/kineticEnergy/KE.csv"
if len(sys.argv) > 1:
    filename = sys.argv[1]

# --- read CSV (no headers) ---
df = pd.read_csv(filename, header=None, names=["time", "KE_total", "KE_volAvg"])

# --- remove duplicate times (keep first entry) ---
df = df.drop_duplicates(subset="time", keep="first")

# --- sort just in case ---
df = df.sort_values("time")

# --- plot ---
plt.figure(figsize=(7,5))
plt.plot(df["time"], df["KE_total"], marker="o", linewidth=1.5)
plt.xlabel("Time [s]")
plt.ylabel("Total kinetic energy [J]")
plt.title("Kinetic Energy vs Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("ke.png", dpi=150)
plt.show()

# --- optional: print summary ---
print(f"Read {len(df)} unique time steps from {filename}")
print(f"Time range: {df['time'].min():.3f} – {df['time'].max():.3f}")
print(f"KE range:   {df['KE_total'].min():.3e} – {df['KE_total'].max():.3e}")
