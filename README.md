# Signal Processing and Learning (SP&L) - Academic Portfolio

This repository contains the full implementation and theoretical analysis of three major Signal Processing and Learning laboratories conducted at Politecnico di Milano.

---

## 📂 Laboratory Details

###  HW1: Parametric Estimation & Statistical Bounds
This lab focuses on the fundamental limits of estimating parameters from noisy complex exponential signals.
* **Core Objectives:**
    * **Maximum Likelihood (ML) & Least Squares (LS):** Implementation of frequency estimators using fine-grid searches on the periodogram.
    * **Cramér-Rao Bound (CRB) Derivation:** Theoretical derivation of the Fisher Information Matrix (FIM) for the unknown parameters: Amplitude $A$, Phase $\phi$, and Frequency $f_0$.
    * **Threshold Effect Analysis:** Investigating the "breakdown" point where SNR is too low for the estimator to maintain efficiency, leading to a rapid increase in MSE.
    * **Asymptotic Efficiency:** Demonstrating that as $N \to \infty$, the estimator variance converges to the CRB.

###  HW2: Adaptive Filtering, System Identification & Kalman Tracking
A multi-scenario study on linear filtering techniques for time-varying and unknown systems.
* **Core Objectives:**
    * **LMS vs. RLS Benchmarking:** Evaluating the trade-off between the simplicity of Least Mean Squares and the fast convergence (at higher computational cost) of Recursive Least Squares.
    * **State-Space Modeling:** Implementation of the **Kalman Filter (KF)** to estimate state variables under Gaussian process and measurement noise.
    * **Active Noise Control (ANC):** A Simulink-based experiment to recover a desired signal from a noisy secondary path using adaptive interference cancellation.
    * **Parameter Sensitivity:** Analysis of how the forgetting factor $\lambda$ (RLS) and step-size $\mu$ (LMS) influence the tracking of non-stationary signals.

###  HW3: Statistical Signal Characterization & Linear Prediction
An exploration of second-order statistics and predictive modeling for stationary random processes.
* **Core Objectives:**
    * **Second-Order Statistics:** Estimation of the **Autocorrelation Function (ACF)** and **Cross-correlation (CCF)** from finite data records to characterize signal dynamics.
    * **Yule-Walker Equations:** Designing optimal **Linear Predictors** by solving the normal equations to find coefficients that minimize the mean squared prediction error.
    * **Parametric Spectral Estimation:** Using AR (Autoregressive) models to provide smoother, higher-resolution PSD estimates compared to the non-parametric Periodogram.
    * **Learning-Based Prediction:** Implementation of a **Deep Neural Network (DNN)** to predict future signal values, with a performance comparison against classical Wiener-based linear predictors.

---
## 🛠️ Tools
* **MATLAB / Simulink:** Primary simulation environment.
* **Statistical Methods:** Bias/Variance analysis, MSE optimization, and Convergence study.

*Professional Coursework - Technology Enhanced Learning (TEL) @ Polimi.*