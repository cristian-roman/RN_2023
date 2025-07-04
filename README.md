```
# RN_2023 – Neural Networks & Learning Systems

This repository contains coursework developed for a class focused on **Neural Networks** and **Machine Learning Fundamentals**. It spans simple learning algorithms like Perceptron and Adaline, progressing toward custom neural networks and applying them in game simulations.

---

## 🗂️ Project Structure

```
RN_2023-main/
├── Tema_1/   # Perceptron & Adaline learning systems
├── Tema2/    # Basic custom neural network implementation
├── Tema3/    # Flappy Bird game controlled by a trained model
```

---

## 🧠 Technologies & Concepts

### ✅ Tema_1 – Perceptron & Adaline
- **adalinePerceptron.py**: Implements the Adaline learning algorithm.
- **adalineLoadingPerceptron.py**: Includes model persistence (save/load).
- **dataUnit.py**: Likely handles sample representation.
- **dataset/**: Contains training/testing data.
- **models/**: Stores trained models or architecture-related components.

### ✅ Tema2 – Custom Neural Network
- **NeuralNetwork.py**: Manual implementation of a basic feedforward network.
- **dataset/**: Input/output sample sets for training.
- Shows how to use neural networks without high-level frameworks.

### ✅ Tema3 – Flappy Bird AI
- **main.py**: Trained neural network agent for playing Flappy Bird.
- **flappy_bird_model.pth**: Pre-trained PyTorch model.
- Demonstrates reinforcement or supervised learning in a game environment.

---

## 🛠️ Stack & Tools

- **Language**: Python 3
- **Libraries**:
  - Likely `numpy` used for numerical computations
  - `torch` (PyTorch) used in Tema3 for model deployment
- No use of high-level ML libraries in early assignments – built from scratch

---

## ▶️ Running the Code

Run each assignment individually:

```bash
# Tema 1 – Perceptron/Adaline
cd Tema_1
python3 main.py

# Tema 2 – Custom Neural Network
cd ../Tema2
python3 main.py

# Tema 3 – Flappy Bird AI
cd ../Tema3
python3 main.py
```

---

## 📌 Notes

- `.idea` and `__pycache__` folders can be ignored.
- This repo is great for learning:
  - How perceptrons and Adaline work
  - Manual neural network implementation
  - Applying learned models to control environments (Flappy Bird)

---

## 🎓 Educational Value

This repository is ideal for students learning:
- Fundamentals of neural network training
- Practical implementation of basic models
- Transition from theory to real-world application

```
