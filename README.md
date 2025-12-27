# OmniNeuro: A Multimodal HCI Framework for Explainable BCI Feedback 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
(https://deepmind.google/technologies/gemini/)

**OmniNeuro** is a novel Brain-Computer Interface (BCI) framework designed to bridge the gap between complex neural decoding and human understanding. 

Unlike traditional "Black Box" BCIs that only output a command, OmniNeuro acts as a **transparent feedback partner**. It utilizes interpretable metrics from **Physics**, **Chaos Theory**, and **Quantum Mechanics** to provide real-time multimodal feedback (Sonification + AI Reports), helping users understand their brain states and improve neuroplasticity.

>  **Paper Reference:** Aghaei Nia, A. (2025). *OmniNeuro: A Multimodal HCI Framework for Explainable BCI Feedback via Generative AI and Sonification*.

---

##  Key Features

OmniNeuro functions as an interpretability layer orthogonal to standard decoders (like EEGNet or CSP). It processes EEG signals through three distinct engines:

### 1. Physics Engine (Thermodynamics)
* **Concept:** Energy Conservation & Entropy.
* **Metric:** Logarithmic Energy Ratio ($L_{idx}$) between C3 and C4 channels.
* **Feedback:** Visualizes the "intensity" and lateralization of mental effort.

### 2. Chaos Engine (Complexity)
* **Concept:** Fractal Geometry.
* **Metric:** Higuchi Fractal Dimension (HFD).
* **Feedback:** Measures neural signal "roughness" to distinguish active computation from resting states or noise.

### 3. Quantum-Inspired Engine (Uncertainty)
* **Concept:** Geometric Probability (Bloch Sphere).
* **Metric:** Probability amplitude ($P_{move} = \sin^2(\theta/2)$).
* **Feedback:** Models decision uncertainty to prevent binary flickering, providing smooth, continuous feedback.

---

## Installation

### Prerequisites
* Python 3.8 or higher
* Google Cloud API Key (for Gemini AI reports)

### 1. Clone the Repository
```bash
git clone [https://github.com/ayda-aghaei/OmniNeuro.git](https://github.com/ayda-aghaei/OmniNeuro.git)
cd OmniNeuro
