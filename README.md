<h1>LCDM: Latent Cognitive Dynamical Model</h1>
<p>
    Welcome to the <strong>LCDM</strong> repository! This project introduces a novel framework for manifold inference and neural decoding,
    specifically designed for analyzing high-dimensional EEG data collected during a word categorization task. The framework combines
    <strong>state-space models (SSM)</strong> with <strong>deep neural networks (DNN)</strong> to infer latent manifolds and predict task-specific labels
    in a unified and interpretable manner.
</p>

<h2>Key Features</h2>
<ul>
    <li><strong>State-Space Modeling with Particle Filters</strong>: Efficient inference of latent states using particle filters to capture the dynamics of high-dimensional neural data.</li>
    <li><strong>Generative-Discriminative Integration</strong>: Combines a generative state evolution model with a discriminative neural network for supervised label prediction.</li>
    <li><strong>1D Convolutional Neural Network (1D CNN)</strong>: Utilizes a 1D CNN to decode latent states into task-specific labels with high precision.</li>
    <li><strong>Expectation-Maximization (EM) Optimization</strong>: Iteratively refines latent states and model parameters for robust manifold inference and label prediction.</li>
    <li><strong>Designed for Cognitive Neuroscience</strong>: Evaluated on EEG data from a word categorization task, distinguishing "Life" vs. "Death" categorizations in participants with and without Major Depressive Disorder (MDD).</li>
</ul>

<h2>Installation</h2>
<p>Clone this repository and install the required dependencies:</p>
<pre><code>git clone https://github.com/&lt;your-username&gt;/lcdm.git
cd lcdm
pip install -r requirements.txt</code></pre>

<h2>Usage</h2>
<h3>Example: Word Categorization Task</h3>
<ol>
    <li><strong>Prepare Your Data:</strong> Format your EEG data as multi-channel time series. Ensure that it is compatible with the input requirements of the model.</li>
    <li><strong>Run the LCDM Model:</strong> Execute the <code>main.py</code> script to infer the latent states and decode task-specific labels:
        <pre><code>python main.py --data_path ./data/eeg_dataset.csv --output_dir ./results</code></pre>
    </li>
    <li><strong>Visualize Results:</strong> Use the provided utilities to analyze and visualize the inferred manifold and decoding performance:
        <pre><code>python visualize.py --results_dir ./results</code></pre>
    </li>
</ol>

<h2>Documentation</h2>
<p>
    Comprehensive documentation for LCDM, including API details, examples, and theory, can be found in the <a href="docs/">docs</a> directory.
</p>

<h2>Examples</h2>
<ul>
    <li><a href="examples/word_categorization_task.ipynb"><strong>Word Categorization Task</strong></a>: Demonstrates how to apply LCDM to EEG data and visualize the inferred latent manifold.</li>
    <li><a href="examples/latent_state_inference.ipynb"><strong>Latent State Inference</strong></a>: Showcases the particle filter and EM algorithm in action.</li>
    <li><a href="examples/performance_metrics.ipynb"><strong>Performance Metrics</strong></a>: Evaluates model performance using accuracy, ROC curves, and AUC.</li>
</ul>

<h2>Benchmarks</h2>
<p><strong>LCDM</strong> was evaluated on EEG data from a cognitive neuroscience study. The results include:</p>
<ul>
    <li><strong>Latent Manifold Representation:</strong> 15-dimensional manifold inferred from high-dimensional EEG data.</li>
    <li><strong>Accuracy:</strong> 85% for "Life" vs. "Death" categorization.</li>
    <li><strong>AUC:</strong> 0.91, indicating excellent discriminative performance.</li>
</ul>

<h2>Citation</h2>
<p>If you use LCDM in your research, please cite us:</p>
<pre><code>@article{lcdm2024,
title={LCDM: Latent Cognitive Dynamical Model for EEG Analysis},
author={Your Name and Collaborators},
journal={arXiv preprint arXiv:XXXX.XXXXX},
year={2024}
}</code></pre>

<h2>Contributing</h2>
<p>We welcome contributions! Please see <a href="CONTRIBUTING.md">CONTRIBUTING.md</a> for guidelines.</p>

<h2>License</h2>
<p>
    This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.
</p>

<h2>Acknowledgments</h2>
<p>
    This work is inspired by advancements in state-space modeling, deep learning, and neuroscience.
    Special thanks to contributors and researchers in the field of cognitive neuroscience and computational modeling.
</p>
