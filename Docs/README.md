<h1>LCDM: Latent Cognitive Dynamical Model</h1>

<h2>Motivation</h2>
<p>
    The study of high-dimensional time series data in neuroscience is essential for understanding the brain's complex mechanisms and functions. Neural recordings, such as EEG, produce vast amounts of data that are challenging to interpret directly due to their high dimensionality and inherent noise. 
    The manifold hypothesis, which suggests that such high-dimensional data resides on a lower-dimensional latent manifold, has become a cornerstone for various applications, including neural mechanism discovery, data visualization, and decoding.
</p>
<p>
    Despite the development of numerous algorithms under this hypothesis, several challenges persist as neural datasets grow and become complex. Current approaches often require multiple disjointed processing steps, limiting their ability to simultaneously infer the underlying manifold and perform tasks like decoding or prediction. Additionally, most methods either lack interpretability or fail to achieve the discriminative power necessary for applications such as label prediction in cognitive tasks.
</p>
<p>
    Addressing these gaps is crucial for neuroscience research, particularly for understanding how neural data evolves across different temporal scales and task conditions. For instance, categorizing emotionally charged words, such as "Life" and "Death," requires models that can work seamlessly across temporal resolutions. This need for a comprehensive and statistically principled framework motivates the development of our novel approach, which integrates state-space models (SSM) with deep neural networks (DNN). By achieving a balance between interpretability and predictive accuracy, our method aims to advance the field by addressing long-standing challenges in manifold inference and neural decoding.
</p>

<h2>Methods</h2>
<p>
    To address the challenges of manifold inference and neural decoding in high-dimensional time series data, we developed a novel framework that combines state-space models (SSM) with deep neural networks (DNN) for supervised manifold learning (see Fig. 1). This approach integrates generative and discriminative modeling, enabling simultaneous manifold inference and label prediction.
</p>

<h3>Latent Dynamical Model</h3>
<p>
    Our framework is based on a latent dynamical model that describes the evolution of the latent state (manifold) and its relationship with neural observations and labels. The model is defined by three main components:
</p>

<h4>State Evolution</h4>
<p>
    The evolution of the latent state X<sub>k</sub> over time is governed by a state equation:
</p>
<pre>
    x_(k+1) | x_k ~ f_ψ(x_k, ϵ_k), ϵ_k ~ N(0, R), x_k ∈ R^D
</pre>
<p>
    Here, ϵ_k ~ N(0, R) represents Gaussian process noise, X<sub>k</sub> ∈ R<sup>D</sup> is the D-dimensional latent state, and f<sub>ψ</sub>(.) is a nonlinear function parameterized by ψ, capturing the dynamics of the manifold.
</p>

<h4>Observation Model</h4>
<p>
    Neural recordings Y<sub>k</sub> are related to the latent state X<sub>k</sub> through an observation equation:
</p>
<pre>
    y_k | x_k ~ g_ϕ(x_k, v_k), v_k ~ N(0, Q), y_k ∈ R^N
</pre>
<p>
    Here, v<sub>k</sub> ~ N(0, Q) represents Gaussian noise, and Y<sub>k</sub> ∈ R<sup>N</sup> is the N-dimensional neural observation. The nonlinear function g<sub>ϕ</sub>(.), parameterized by ϕ, maps the latent state to the observed data.
</p>

<h4>Label Prediction</h4>
<p>
    Labels l<sup>t</sup>, such as movement direction, are modeled as a function of the entire latent trajectory:
</p>
<pre>
    l | x_(0…K) ~ h_ϕ(x_(0…K)), l ∈ {0,1}
</pre>
<p>
    Here, l<sup>t</sup> ∈ {0,1}, and h<sub>ϕ</sub>(.) represents a discriminative function parameterized by ϕ. This component allows for supervised learning by linking the inferred manifold to task-specific labels.
</p>

<h3>Training and Inference</h3>
<p>
    The model is trained to maximize a joint likelihood that incorporates both the generative (state evolution and observation models) and discriminative (label prediction) components. This is achieved using a variational approach where the posterior over the latent states is inferred jointly with the model parameters. Key steps include:
</p>
<ul>
    <li><strong>Manifold Inference:</strong> Inferring the latent state X<sub>k</sub> that best represents the high-dimensional data while respecting the temporal dynamics imposed by f<sub>ψ</sub>(.).</li>
    <li><strong>Supervised Learning:</strong> Optimizing h<sub>ϕ</sub> to predict labels l based on the latent trajectory X<sub>0</sub> … X<sub>k</sub>.</li>
</ul>

<h1>Solution</h1>
<p>
    Our proposed framework addresses the challenges of manifold inference and neural decoding through a unified approach that integrates generative and discriminative models. This section details the two core aspects of the solution: Inference and Decoding.
</p>

<h2>Inference</h2>
<p>
    The inference process focuses on uncovering the latent manifold that best represents the high-dimensional EEG data while respecting the temporal dynamics. The latent states <i>X<sub>k</sub></i> are estimated using the state evolution equation, and their alignment with the observations <i>Y<sub>k</sub></i> is achieved through the observation model.
</p>
<p>
    To optimize the manifold inference, we employ an Expectation-Maximization (EM) algorithm in conjunction with a particle filter. The EM algorithm iteratively refines the model parameters and latent state estimates by alternating between the following steps:
</p>
<h3>E-Step (Expectation)</h3>
<p>
    In this step, the posterior distribution of the latent states is approximated given the current parameters of the state-space model. The particle filter is used to estimate this posterior by representing it with a set of weighted particles. Each particle corresponds to a potential realization of the latent state, and the weights reflect the likelihood of each particle given the observed data:
</p>
<pre>w<sub>k</sub><sup>(i)</sup> ∝ P(y<sub>k</sub> | x<sub>k</sub><sup>(i)</sup>)</pre>

<h3>M-Step (Maximization)</h3>
<p>
    The model parameters <i>ψ</i> (for state evolution) and <i>ϕ</i> (for the observation model) are updated by maximizing the expected joint likelihood of the latent states and observations, based on the posterior distribution estimated in the E-step:
</p>
<pre>arg max E<sub>P(x<sub>k</sub> | y<sub>k</sub>)</sub>[log P(y<sub>k</sub>, x<sub>k</sub> | ψ, ϕ)]</pre>
<p>
    The integration of the particle filter within the E-step allows for efficient handling of nonlinear and non-Gaussian dynamics, which are common in neural data.
</p>
<p>
    This EM-based approach ensures that both the latent states and the model parameters converge to values that best represent the data while respecting temporal dynamics. The particle filter facilitates robust inference by focusing computational resources on the most likely regions of the state space.
</p>

<h2>Decoding</h2>
<p>
    The decoding process utilizes the inferred manifold to predict task-specific labels, such as "Life" or "Death" categorizations. The framework incorporates a supervised learning approach, integrating the inferred latent states with a 1D Convolutional Neural Network (1D CNN) to map the latent trajectory <i>X<sub>0</sub> … X<sub>k</sub></i> to labels.
</p>
<h3>Key Features of the Decoding Process</h3>
<ul>
    <li>
        <strong>Label Prediction Model:</strong>
        The 1D CNN is trained to classify the trajectory of latent states into task-specific categories, optimizing for cross-entropy loss to maximize classification accuracy.
    </li>
    <li>
        <strong>Joint Inference and Decoding:</strong>
        The EM algorithm ensures that decoding benefits directly from the inferred manifold, as both steps are integrated into a unified framework.
    </li>
</ul>
<p>
    By combining the EM algorithm for parameter optimization, the particle filter for robust manifold inference, and the 1D CNN for decoding, the proposed framework achieves high precision in label prediction. This approach provides a powerful tool for modeling and understanding cognitive processes captured in EEG data.
</p>
