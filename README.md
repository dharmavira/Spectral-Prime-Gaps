# kappa_m0_bound.py

## Overview
kappa_m0_bound.py implements the numerical framework for evaluating:
- The alignment constant κ from Selberg sieve quadratic forms.
- The Paley–Zygmund ratio R_0 for baseline prime–pair detectors.
- The control–variate correlation parameter ρ² from variance–drop computations.

These calculations are part of the methodology in Spectral Control–Variates and Bounded Prime Gaps, which combines Maynard/Polymath8 prime gap arguments with variance reduction via an explicit control–variate projection.

The script supports both single–window (conservative) and two–window + primorial thinning (liberal) configurations, allowing reproduction of bounds such as:
- Conservative: m₀ ≤ 242
- Liberal: m₀ ≤ 238
- Exploratory: toward m₀ ≤ 236

## Features
- Explicit κ computation  
  Evaluates the Selberg–kernel quadratic forms and arithmetic progression bump functional directly.
- Prime sum evaluation  
  Computes the sums in ρ² for flat and optimized prime weights.
- Scenario testing  
  Compares single–window, two–window, and exploratory coefficient sets.
- Baseline → Improved bound translation  
  Uses the Paley–Zygmund homogeneity principle to convert variance drop into an improved M (gap) bound.

## Usage
Run the script directly in Python 3:
```bash
python kappa_m0_bound.py
