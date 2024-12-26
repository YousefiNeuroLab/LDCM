# LCDM: Latent Cognitive Dynamical Model

## Motivation
The study of high-dimensional time series data in neuroscience is essential for understanding the brain's complex mechanisms and functions. Neural recordings, such as EEG, produce vast amounts of data that are challenging to interpret directly due to their high dimensionality and inherent noise. 

The manifold hypothesis, which suggests that such high-dimensional data resides on a lower-dimensional latent manifold, has become a cornerstone for various applications, including neural mechanism discovery, data visualization, and decoding.

Despite the development of numerous algorithms under this hypothesis, several challenges persist as neural datasets grow and become complex. Current approaches often require multiple disjointed processing steps, limiting their ability to simultaneously infer the underlying manifold and perform tasks like decoding or prediction. Additionally, most methods either lack interpretability or fail to achieve the discriminative power necessary for applications such as label prediction in cognitive tasks.

Addressing these gaps is crucial for neuroscience research, particularly for understanding how neural data evolves across different temporal scales and task conditions. For instance, categorizing emotionally charged words, such as "Life" and "Death," requires models that can work seamlessly across temporal resolutions. This need for a comprehensive and statistically principled framework motivates the development of our novel approach, which integrates state-space models (SSM) with deep neural networks (DNN). By achieving a balance between interpretability and predictive accuracy, our method aims to advance the field by addressing long-standing challenges in manifold inference and neural decoding.

---

## Methods
To address the challenges of manifold inference and neural decoding in high-dimensional time series data, we developed a novel framework that combines state-space models (SSM) with deep neural networks (DNN) for supervised manifold learning. This approach integrates generative and discriminative modeling, enabling simultaneous manifold inference and label prediction.

### Latent Dynamical Model
Our framework is based on a latent dynamical model that describes the evolution of the latent state (manifold) and its relationship with neural observations and labels. The model is defined by three main components:

#### State Evolution
The evolution of the latent state \( X_k \) over time is governed by a state equation:

\[
x_{k+1} \mid x_k \sim f_\psi(x_k, \epsilon_k), \quad \epsilon_k \sim N(0, R), \quad x_k \in \mathbb{R}^D
\]

Here, \( \epsilon_k \sim N(0, R) \) represents Gaussian process noise, \( X_k \in \mathbb{R}^D \) is the \( D \)-dimensional latent state, and \( f_\psi(\cdot) \) is a nonlinear function parameterized by \( \psi \), capturing the dynamics of the manifold.

#### Observation Model
Neural recordings \( Y_k \) are related to the latent state \( X_k \) through an observation equation:

\[
y_k \mid x_k \sim g_\phi(x_k, v_k), \quad v_k \sim N(0, Q), \quad y_k \in \mathbb{R}^N
\]

Here, \( v_k \sim N(0, Q) \) represents Gaussian noise, and \( Y_k \in \mathbb{R}^N \) is the \( N \)-dimensional neural observation. The nonlinear function \( g_\phi(\cdot) \), parameterized by \( \phi \), maps the latent state to the observed data.

#### Label Prediction
Labels \( l^t \), such as movement direction, are modeled as a function of the entire latent trajectory:

\[
l \mid x_{0…K} \sim h_\phi(x_{0…K}), \quad l \in \{0, 1\}
\]

Here, \( l^t \in \{0, 1\} \), and \( h_\phi(\cdot) \) represents a discriminative function parameterized by \( \phi \). This component allows for supervised learning by linking the inferred manifold to task-specific labels.

---

### Training and Inference
The model is trained to maximize a joint likelihood that incorporates both the generative (state evolution and observation models) and discriminative (label prediction) components. This is achieved using a variational approach where the posterior over the latent states is inferred jointly with the model parameters. Key steps include:

- **Manifold Inference:** Inferring the latent state \( X_k \) that best represents the high-dimensional data while respecting the temporal dynamics imposed by \( f_\psi(\cdot) \).
- **Supervised Learning:** Optimizing \( h_\phi \) to predict labels \( l \) based on the latent trajectory \( X_0 … X_k \).

---

## Solution
Our proposed framework addresses the challenges of manifold inference and neural decoding through a unified approach that integrates generative and discriminative models.

### Inference
The inference process focuses on uncovering the latent manifold that best represents the high-dimensional EEG data while respecting the temporal dynamics. The latent states \( X_k \) are estimated using the state evolution equation, and their alignment with the observations \( Y_k \) is achieved through the observation model.

To optimize the manifold inference, we employ an Expectation-Maximization (EM) algorithm in conjunction with a particle filter. Key steps include:

- **E-Step (Expectation):**
  Approximating the posterior distribution of the latent states using a particle filter:
  \[
  w_k^{(i)} \propto P(y_k \mid x_k^{(i)})
  \]

- **M-Step (Maximization):**
  Updating model parameters \( \psi \) and \( \phi \) by maximizing the expected joint likelihood:
  \[
  \arg\max E_{P(x_k \mid y_k)}[\log P(y_k, x_k \mid \psi, \phi)]
  \]

### Decoding
The decoding process utilizes the inferred manifold to predict task-specific labels, such as "Life" or "Death" categorizations. Key features include:

- **Label Prediction Model:** A 1D Convolutional Neural Network (1D CNN) trained to classify latent state trajectories into task-specific categories.
- **Joint Inference and Decoding:** The EM algorithm ensures decoding benefits directly from the inferred manifold.

By integrating manifold inference, parameter optimization, and decoding in a unified framework, this approach provides a powerful tool for modeling and understanding cognitive processes in EEG data.
