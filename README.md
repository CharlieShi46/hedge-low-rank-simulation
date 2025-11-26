# hedge-low-rank-simulation
Faster simulation of Hedge algorithm under low-rank expert structures. Includes code and PDF.
# Faster Simulation of Hedge in Low-Rank Expert Settings

This repository contains the code and supplementary material for the research note:

**“Faster Simulation of Hedge under Low-Rank Expert Structures”**  
Author: Cheng Shi, Michael Spece
2021

## 🧠 Project Overview
This project studies how the Exponential Weights/Hedge algorithm behaves under
low-rank expert-loss matrices. Prior work required slow rejection sampling to
generate rank-r matrices. I implemented a faster constructive method that:

- generates exact low-rank binary loss matrices efficiently  
- enables large-scale simulation (1M+ runs)  
- allows empirical measurement of worst-case regret as rank varies  

The experiments confirm that Hedge **implicitly adapts to low-rank structure** and
exhibits the expected √r regret behavior.

## 📂 Code Structure
