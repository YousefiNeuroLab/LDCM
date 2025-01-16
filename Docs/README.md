# Robust DNN-based Decoder Model with an Embedded State-Space Model Layer

## Authors
- **Pedram Rajaei**  
  Department of Biomedical Engineering, University of Houston, TX, USA  
  Email: prajaei@CougarNet.uh.edu  
- **Ali Yousefi**  
  Department of Biomedical Engineering, University of Houston, TX, USA  
  Email: aliyousefi@uh.edu  

---

## Abstract
A critical step in probing brain function is characterizing time-series data representing biobehavioral signals recorded under various conditions or cognitive tasks. These data are complex, volatile, and stochastic, posing significant modeling challenges. While deep neural networks (DNNs) have demonstrated effectiveness in characterizing such data, they are sensitive to noise and require large datasets for training. This research introduces a novel framework integrating state-space models (SSMs) into the DNN structure, addressing these limitations. The resulting model, SSM-DNN, achieves superior performance metrics (specificity, sensitivity, AUC) and surpasses state-of-the-art DNN models in decoding participants' phenotypes (e.g., Major Depressive Disorder vs. Healthy). This approach is applicable across a broader class of neuroscience data.

---

## Keywords
- Neural Decoder
- Bayesian Inference
- Adaptive Dimensionality Reduction
- Supervised State-Space Model
- Deep Neural Network (DNN)

---

## Introduction
Deep neural networks (DNNs), including CNNs and RNNs, have revolutionized neuroscience by achieving state-of-the-art performance in decoding EEG and fMRI data. However, the need for large-scale, high-quality labeled datasets and the presence of noise in neuroscience data limit their utility. The proposed **SSM-DNN** addresses these challenges by integrating the adaptive noise reduction capabilities of SSMs with the expressive and discriminative power of DNNs. This hybrid model effectively manages data noise and size limitations, offering a robust solution for high-dimensional time-series data analysis.

---

## Methods

### [Model Definition](Documentation.pdf)
SSM-DNN combines latent dynamical manifolds from SSMs with supervised learning tasks of DNNs. It includes:
- A **latent variable** capturing essential dynamics from biobehavioral signals.
- State and observation models represented mathematically.
- Training via Expectation-Maximization (EM) combined with MCMC sampling for latent state inference.
    <img src="https://github.com/YousefiNeuroLab/LDCM/blob/main/Pictures/Model-Structure.png" alt="SSN-DNN" height="300" width="360">
    <figcaption><strong>SMM-DNN Model Architecture</strong> The model combines SSM and DNN in characterization of high-dimensional neural recording and their labels.  represent a low dimensional representation of neural data, which is passed to a DNN for supervised learning tasks, enabling accurate label prediction and interpretation. This integration leverages the generative strengths of SSMs for capturing temporal dynamics while utilizing the discriminative capabilities of DNNs for classification and regression.
    </figcaption>

### Training
The model training involves:
1. **E-step:** Posterior distribution estimation using MCMC.
2. **M-step:** Updating parameters with stochastic gradient ascent.
<img src="https://github.com/YousefiNeuroLab/LDCM/blob/main/Pictures/EM-algorithm.png" alt="EM" height="300" width="360">
    <figcaption><strong>Expectation-Maximization (EM) Algorithm for Maximum Likelihood Estimation</strong>
The graph illustrates how the EM algorithm converges to a local maximum by iteratively performing the E-step and M-step.
    </figcaption>

### Decoding
The [decoding process](API.md) predicts the label probability per trial using state inference derived from observed data.

---

## Data Description
The [dataset](https://github.com/YousefiNeuroLab/LDCM/blob/main/Data%20Description.md) includes behavioral data from an **Implicit Association Task (IAT)** designed to distinguish between individuals with Major Depressive Disorder (MDD) and healthy controls (CTL). Key details:
- **Participants:** 23 (11 MDD, 12 CTL).
- **Task:** Matching stimulus words with categories ("Life + Me" or "Death + Me").
- **Trials:** 360 trials per participant across 18 blocks.
- **Data:** Reaction times (RT) and accuracy were recorded to analyze phenotypes (MDD vs. CTL).

---

## Performance Results
The SSM-DNN outperformed traditional DNNs, achieving higher specificity, sensitivity, and AUC scores. A comparison with transfer learning models revealed significant performance improvements due to domain-specific training and noise resilience.

---

## Future Work
Future directions include integrating EEG signals with behavioral data to enhance decoding performance and generalizability to multivariate time-series datasets.

---

## Acknowledgments
This project is sponsored by the **Defense Advanced Research Projects Agency (DARPA)** under cooperative agreement No. N660012324016. The content reflects the authors' views and does not necessarily represent official policies of the government.

---

## References
You can find a draft of our working paper [here](Documentation.pdf).

---

## Contact
For further information, please contact:
- Pedram Rajaei: prajaei@CougarNet.uh.edu
- Ali Yousefi: aliyousefi@uh.edu
