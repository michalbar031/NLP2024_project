# Multi-Armed Bandit Extension for: "Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation"
<img src="https://github.com/michalbar031/NLP2024_project/assets/81368958/31da3edc-32db-40b1-98cc-988b0f27b3ac" alt="Project Image" width="360" height="200" align="right">


In this project, we will try to advance the research in predicting binary actions of players in language-based persuasion games by improving the data simulation. First, we propose to enrich the data production phase by incorporating an additional strategy into the simulation process. Our new strategy aims to generate more diversified and realistic scenarios that better reflect human decision-making dynamics, thus enhancing the robustness and applicability of our predictive models.

Second, we intend to refine the strategy selection and enhance the learning process to resemble real human decision-makers. By implementing a Multi-Armed Bandit algorithm into the strategy selection we are trying to get more realistic simulated data.

## The MAB Extension: Our Simulation Setup
<img src="https://github.com/michalbar031/NLP2024_project/assets/81368958/0d3646e5-2506-49c6-831d-f0f6683dc38b" alt="Project Image" width="360" height="200" align="right">


In our simulation, the arms corresponded to different strategies that the decision-making bots could employ. Each strategy was designed to reflect various decision-making behaviors observed in humans. We simulated interactions between bots and human-like agents over a series of rounds. Each round involved the bots selecting reviews for hotels, and the agents making decisions based on these reviews. Instead of using a fixed probabilistic vector to choose strategies, we applied the MAB framework. Specifically, we used Thompson Sampling to dynamically select the most promising strategies based on their performance. The goal was to maximize the cumulative reward over multiple rounds of interaction, simulating realistic human-machine decision-making processes.
Furthermore, we implemented two new human-like behavior strategy functions. We created the following strategies: A classifier-based strategy, and a cost-benefit hybrid strategy. As well as using the Trustful Strategy, Language-Based Strategy, and Random Strategy. 


## Getting Started

### Prerequisites

Before you begin, ensure you have the following tools installed on your system:
- Git
- Anaconda or Miniconda

### Installation

To install and run the code on your local machine, follow these steps:

1. **Clone the repository**

   First, clone the repository to your local machine using Git. Open a terminal and run the following command:
   ```bash
   git clone https://github.com/michalbar031/NLP2024_project.git
    ```
2. **Create and activate the conda environment**

    After cloning the repository, navigate to the project directory:

    ```bash
    cd NLP2024_project
    ```

    Then, use the following command to create a conda environment from the requirements.yml file provided in the project:
    ```bash
    conda env create -f requirements.yml
    ```
3. **Log in to Weights & Biases (W&B)**

   Weights & Biases is a machine-learning platform that helps you track your experiments, visualize data, and share your findings. Logging in to W&B is essential for tracking the experiments in this project. If you haven't already, you'll need to create a W&B account. 
   Use the following command to log in to your account:
    ```bash
    wandb login
    ```


This `README.md` file provides clear instructions for setting up the project on a local machine, ensuring that all necessary tools and dependencies are installed correctly. If you have additional sections or content you would like to add, please let me know!
