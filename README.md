# Aerodynamic-Optimisation
Machine-learning based optimization of rear-end vehicle geometry using KNN regression and numerical optimization to reduce aerodynamic drag

# Data-Driven Aerodynamic Optimisation of Rear Vehicle Geometry

This project explores whether machine learning can be used as a faster and practical alternative to repeated CFD simulations when optimising the rear-end geometry of a vehicle.

## ✨ Goal

To reduce aerodynamic drag (and improve lift-to-drag ratio) by predicting aerodynamic performance using a regression model and automatically searching for the optimal diffuser and ramp angles.

## 🧠 Method

- Used the **DrivAerNet Parametric Dataset** (4,000+ CFD simulations of different car geometries)
- Computed **lift-to-drag ratio (L/D)** from front lift and drag coefficients
- Trained and compared 4 regression models:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbours (KNN)
- Tuned hyperparameters with **GridSearchCV**
- Selected best model (KNN) and applied **numerical optimisation** (SciPy) to find the angles which maximise aerodynamic efficiency
- Visualised results with a **heatmap** and **3D surface plot**

## 🔧 Files

| File | Description |
|------|-------------|
| `drivaernet_optimisation.py` | Main Python script (data loading, modelling, optimisation) |
| `DrivAerNet_ParametricData.csv` | Dataset used for training and optimisation |
| `README.md` | This file |

## 📈 Output

- **Best ramp angle** found:  *≈ 12–13°*
- **Best diffuser angle** found:  *≈ 11–12°*
- Significant improvement in lift-to-drag ratio compared to average dataset value

## 📌 Future Work

- Include more geometry features (e.g. spoiler or roof curvature)
- Validate ML predictions with wind-tunnel or higher-fidelity CFD
- Extend to **multi-objective optimisation** (drag + downforce)

---

If you use this project or have any suggestions, feel free to open an issue or pull request!
