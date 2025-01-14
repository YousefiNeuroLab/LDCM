<h1>Latent-state Dynamical Coherence Model (LDCM)</h1>
<img src="Pictures/Logo_LDCM.png" alt="LCDM Picture" height="320" width="320">
<p>
    <mark>This will be the repository for LDCM project. The material in this repository 
    is based on a current research which combines SSM and DNN for 
    decoding high dimensional neural data. We designed the content of this 
    repository aligned with what we thought it would be for LDCM project. Part of 
    our previous effort on LDCM such as SS-GCoh model and preliminary 
    results on the anesthesia data can be found in our previous GitHub 
    repository developed for this research: (https://github.com/YousefiNeuroLab/LDCM)</mark>
</p>
<p>
    This project introduces a novel 
    framework for manifold inference and neural decoding solution, specifically 
    designed for analysis of high-dimensional data 
    collected during cognitive tasks. In this 
    framework, we combine <strong>state-space models 
    (SSM)</strong> with <strong>deep neural networks (DNN)</strong>, in our effort to characterize 
    high dimensional data, and also infer latent dynamical manifolds, which 
    capture essential dynamics present in data and associated condition or 
    label. For the application, we show the whole modeling pipeline in a 
    Implicit Association Task which is called brief death IAT, recorded in our 
    research group under approved IRB. Details of this task can be found <a href="Data Description.md" target="_blank">here.</a>
</p>

<h2>Key Features</h2>
<p>Development of SSM-DNN involves development of multiple modeling 
techniques which include:</p>
<ul>
    <li><strong>MCMC sampling technique ( Particle Filters) for SSM-DNN model:</strong>: Efficient inference solution of latent states, which turns into  particle filters solution deriving an approximate posterior distribution of state given neural data and 
associated labels.</li>
    <li><strong>Data Generation Pipeline</strong>: Generative nature of SSM-DNN model which emerges from its SSM 
element lets us to draw samples (trajectories) of the high dimensional 
data and corresponding labels or categories.</li>
    <li><strong>Flexible DNN Topologies Embedded in SSM-DNN</strong>: The SSM-DNN training and 
inference are agnostic to the DNN topology and structure; thus, we 
demonstrate how model can be applied to Multi-Layer Perceptron Neural Network, CNN with 1-D input and also CNN with multivariate time 
series.</li>
    <li><strong>Verstatile Learning Solution</strong>: Expectation-Maximization (EM) 
based training solution combined with sampling technique and 
stochastic gradient techniques are built for the SSM and DNN model 
training</li>
    <li><strong>Flexibility in Analysis of Different Modalities of Data</strong>: The framework can be applied to different modalities of data 
beyond neural data, and it can process behavioral time-series data or 
mixture of behavioral signals.</li>
</ul>

<h2>Installation</h2>
<p>Clone this repository and install the required dependencies:</p>
<pre><code>git clone https://github.com/&lt;your-username&gt;/ldCm.git
cd ldCm
pip install -r requirements.txt</code></pre>

<h2>Usage</h2>
<p>To use this toolkit, follow the step-by-step instructions provided below to set up, train, and evaluate the SSM-DNN model for your high-dimensional data analysis tasks.</p>
<ol>
    <li>
        <strong>Prepare Your Data:</strong> Format your data as multi-channel time series. Ensure that it is 
        compatible with the input requirements of the model.
    </li>
    <li>
        <strong>Run the SSM-DNN Model:</strong> Execute the <code>main.py</code> script to infer the latent states 
        and decode task-specific labels:
        <pre><code>python main.py --data_path ./data/eeg_dataset.csv --output_dir ./results</code></pre>
    </li>
    <li>
        <strong>Visualize Results:</strong> Use the provided utilities to analyze and visualize the inferred manifold 
        and decoding performance:
        <pre><code>python visualize.py --results_dir ./results</code></pre>
    </li>
    <li>
        <strong>Run on Google Colab:</strong> Use the pre-configured notebook for an interactive experience: 
        <a href="https://colab.research.google.com" target="_blank">Run on Colab</a>
    </li>
</ol>

<h2>Modeling approach and definition</h2>
<p>
    Comprehensive documentation for SSM-DNN, including API details, examples, and theory, can be found in the <a href="Docs/">docs</a> directory.
</p>
<h2>Dataset Repository</h2>

<p>You will find the dataset used in this research, along with datasets for upcoming work, on the <a href="https://dandiarchive.org/ target="_blank">DANDI archive</a>. The datasets shared on DANDI comply with NIH regulatory guidelines and have been approved by local Institutional Review Boards (IRBs). All datasets have been de-identified to ensure compliance with ethical considerations and HIPAA regulations.</p>
<ul>
    <li><a href="https://dandiarchive.org/dandiset/001285?pos=1" target="_blank>EEG Anesthesia Dataset:</a> This dataset is being recorded as part of the research conducted in this study. It includes EEG data collected during anesthesia experiments to support the development and evaluation of the proposed models. The dataset adheres to ethical guidelines, with all necessary approvals obtained from relevant Institutional Review Boards (IRBs). The data is de-identified to ensure compliance with privacy standards and HIPAA regulations.</li>
</ul>
    
<h2>Examples</h2>

<article id="example-1">
<h3>I. Brief Death Implicit Association Task (BDIT)</h3>
<p>
  We validate our approach using behavioral data recorded during an Implicit Association Test (IAT),
  where participants classify labels (e.g., MDD vs. CTRL). Our results demonstrate that the framework
  achieves superior accuracy, sensitivity, and specificity compared to existing methods. This contribution
  addresses the critical need for approaches that balance interpretability with predictive power, making our
  framework particularly valuable for analyzing biological and neural data.
</p>
<h4>Explore Code:</h4>
<ul>
    <li><a href="Neural_Network_for_Classification_Task.py">Neural Network for Classification Task</a>: Demonstrates how to implement and train a neural network for EEG data classification and visualize its performance.
    </li>
    <li>
        <a href="LatentStateInference.py">Latent State Inference</a>: Showcases the particle filter and EM algorithm in action.
    </li>
    <li>
        <a href="PerformanceMetrics.py">Performance Metrics</a>: Evaluates model performance using accuracy, ROC curves, and AUC.
    </li>
</ul>
</article>

<article id="example-2">
<h3>II. Simulated Observation Dataset Classification</h3>
<p>
  In this example, we create a Simulated dataset with the following characteristics:
</p>
<ul>
  <li><strong>Observation Dimension:</strong> 2</li>
  <li><strong>Latent State Dimension:</strong> 2</li>
  <li><strong>Trials:</strong> 400</li>
  <li><strong>Time Steps per Trial:</strong> 25</li>
</ul>
<p>
  We applied our framework using <strong>4000 particles</strong> to infer the classification results.
  The 1D Convolutional Neural Network (CNN) used in this example consists of <strong>two convolutional layers</strong>
  and is optimized with the <strong>ADAM optimizer</strong>. This example demonstrates how the model handles
  time-series data in scenarios with predefined dimensionalities and trial structures.
</p>
<h4>Explore Code:</h4>
<ul>
  <li>
    <a href="https://colab.research.google.com/drive/1-F3BZ0BUh1HzCy5zRQy911UAcI7O6T8q#scrollTo=sGYr5ho1iHdu/" target="_blank">
      Colab Notebook for Simulated Observation Dataset
    </a>
  </li>
</ul>


<h2>Citation</h2>
<p>If you use SSM-DNN in your research, please cite us:</p>
<pre><code>@article{LDCM2024,
title={LDCM: Latent Cognitive Dynamical Model for EEG Analysis},
author={Your Name and Collaborators},
journal={arXiv preprint arXiv:XXXX.XXXXX},
year={2024}
}</code></pre>

<h2>Collaboration and Contribution</h2>
<p>We welcome contributions! Please see <a href="CONTRIBUTING.md">CONTRIBUTING.md</a> for guidelines.</p>

<h2>License</h2>
<p>
    This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.
</p>

<h2>Acknowledgments</h2>
<p>
    This work was partially supported by (add DARPA funding here). We 
appreciate our research collaboratros from UMN, Intheon, and etc
The work was also supported by start up fund from University of 
Hosuton. Our special thanks go to our research collaborator and 
colleague who contributed in provifing data and thoughtful comments 
in revising our modeling framework.
</p>
