# Advanced Machine Learning & AI — Neural Network Training Workflows

🎓 **Course:** DX703 – Advanced Machine Learning & AI (Boston University, OMDS)  
👩🏻‍💻 **Author:** Tetiana Kravchuk  
📍 **Focus:** Neural network training, regularization, and model selection

---

## 📌 Overview

This repository contains hands-on machine learning assignments focused on **training deep neural networks effectively** — not just building models, but understanding *how* and *why* they learn.

The work emphasizes:
- Monitoring training using **learning curves**
- Preventing overfitting with **early stopping**
- Systematic tuning of **activation functions, learning rates, dropout, and L2 regularization**
- Designing a **best-performing model** using experimental evidence

All experiments are conducted on the **Forest Cover Type (Covertype)** dataset, a large multi-class tabular benchmark.

---

## 🧠 Key Skills Demonstrated

- Neural network design with **TensorFlow / Keras**
- Multi-class classification on real-world tabular data
- Early stopping & validation-based model selection
- Regularization techniques:
  - Dropout
  - L2 (weight decay)
- Learning rate optimization & scheduling
- Reproducible experimentation workflows
- Clear interpretation of training vs. validation behavior

---

## 📂 Repository Contents

### ⭐ Homework 03 — Learning Curves & Training Workflow
📍 `homework/homework-03/Homework_03.ipynb`

This is the **main highlight** of the repository.

It includes:
- Baseline neural network design
- Activation function comparison (ReLU, sigmoid, tanh)
- Learning rate sweeps
- Dropout experiments
- L2 regularization experiments
- Combined regularization strategies
- Final “best model” with learning rate scheduling

Each model is selected using **early stopping at minimum validation loss**, mirroring real-world ML practice.

---

### Homework 02 — Neural Network Foundations
📍 `homework/homework-02/Homework_02.ipynb`

Earlier assignment covering:
- Multi-class neural networks
- Gradient descent behavior
- Baseline performance evaluation

---

## 📊 Dataset

**Forest Cover Type Dataset (Covertype)**  
- ~581,000 samples
- 54 cartographic features
- 7 forest cover classes
- Balanced subset used for controlled experimentation

---

## 🧪 Experimental Philosophy

This repository prioritizes:
- **Validation accuracy**, not training accuracy
- **Generalization**, not memorization
- **Evidence-based decisions**, not guesswork

Models are evaluated at the epoch of **minimum validation loss**, not the final epoch.

---

## 🚀 How to Explore

1. Start with **Homework 03**
2. Scroll through training curves & validation metrics
3. Review how each hyperparameter choice affects generalization
4. Examine the final model design and rationale

---

## 🔧 Tech Stack

- Python
- NumPy, Pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

---

## 📬 Contact

If you’d like to discuss this work or similar ML projects:

- 🌐 Portfolio: *coming soon*
- 💼 LinkedIn: *add link here*
