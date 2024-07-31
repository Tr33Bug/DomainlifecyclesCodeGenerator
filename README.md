# DomainlifecyclesCodeGenerator
The *Domainlifecycles Code Generator (DCG)* formerly *NitroX Code Generator (NCG)* is the first version of a Generative AI Assistance System which was developed for esentri's Domain Lifecycles Framework for the Domain-Driven Design (DDD) Development process.

This first version of the DCG is able to create syntactically correct Domainlifecycles JSON objects as part of the Domainlifecycles DSL. More information about the DCG and its creation, as well as limitations and future work, can be found in the [Master Thesis PDF](Master-Thesis_Götz-Henrik_Wiegand_2024.pdf). 

The DCG was developed in collaboration with [esentri](https://esentri.com/) as part of my Master's thesis at the [Karlsruhe University of Applied Sciences](https://www.h-ka.de/). 

This repository contains the project files, models and logs for the DCG.

Outsourced from this repository is the [DCG-DemoApp](https://github.com/Tr33Bug/DCG-DemoApp) which was developed for the presentation and showcase of the possibilities of this first model. 

> [!IMPORTANT]  
> At the time of the creation of the thesis and the project, the domain lifecycles framework still had the working title NitroX. In the following project, as well as in the entire master thesis, Domainlifecycles is always referred to as NitroX. 

## Overview

```bash
DomainlifecyclesCodeGenerator
├── all_json
│   ├── nitrox.dlc.domain.types.base.AggregateRootBase.json
│   ...
│   └── tests.shared.validation.javax.ValidatedValueObject.json
├── datasets
├── gen_json
│   │   ├── completed
│   │   │   ├── 0.json
│   │   │   ...
│   │   │   └── 99.json
│   ├── 1_0_Part1_NitroXObjects_EntityModel.json
│   ...
│   ├── 2_49_part2_riskyPrompt.json
│   ├── parsibility_results.csv
│   └── parsibility_results.txt
├── models
│   └── finalTraining_v1
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── generation_config.json
│   │   └── training_args.bin
├── runs
│   └── May22_23-13-35_deep-learning_FinalTraining_v1
│   │   └── events.out.tfevents.1716412415.deep-learning.1381720.0
├── .gitignore
├── 1_datasetGenerator.ipynb
├── 2_trainingLoop.ipynb
├── 3_hyperparameterTrainer.ipynb
├── 4_optunaEvaluation.ipynb
├── 5_finalTraining.ipynb
├── ColoredDataPreprocessingProcess.jpg
├── environment.yml
├── LICENSE
├── Master-Thesis_Götz-Henrik_Wiegand_2024.pdf
└── README.md
```

### Folder Structure
- **all_json:** 
    - Folder containing the raw JSON files for fine-tuning the DCG. 
    - The customer-related project data from the "esentri-Partner" was removed from the data set. That was 80% of the files and so the data set is only stored here as an example.
- **datasets:**
    - Empty folder reserved for export and storage of the generated and cleaned dataset.
- **gen_json:**
    - Folder with the generated samples for the Model Assessment phase.
- **models:**
    - Reserved path for the model export with the `finalTraining_v1` model as the result of the final training for the DCG. 
- **runs:**
    - Reserved tensorboard callback folder for the training history logs. The results from the final training of the DCG are stored here. 
    - The metrics and progressions logged there can be displayed and analyzed with a tensorboard.

## System Requirements ###

> [!IMPORTANT] 
> The entire project was developed on a Linux 64-bit system with an NVIDIA graphics card. The setup and the README have therefore only been tested for these specifications: 
> - Ubuntu 22.04.4 LTS
> - GeForce RTX 2080 Ti (11GB VRAM)


## Installation and Setup

1. **Clone the Repository**  

   Clone the repository:
   ```bash
   git clone git@github.com:Tr33Bug/DomainlifecyclesCodeGenerator.git
   cd DomainlifecyclesCodeGenerator
   ```

2. **Install the Requirements**

   Create a conda environment and install all the requirements from the `environment.yml`:
   ```bash
   # create environment
   conda env create -f environment.yml

   # activate environment
   conda activate DCGServerEnv
   ```
3. **Run the Notebooks**

    Start with [1_datasetGenerator.ipynb](1_datasetGenerator.ipynb) and follow the instructions from the jupyter notebook.

## Workflow and Engineering Documentation
This section documents some of the project workflows and their setup. 

### Remote Training
> [!NOTE] 
> The entire project was engineered remotely via VS Code SSH access. In order to be able to close the notebook during longer training times, the notebooks were exported as a Python script and executed remotely with a tmux session. 

The setup and procedure is explained in this section using the example of the 1_datasetGenerator and the 2_trainingLoop notebook:

1. Start session and setup:
    - Start ssh session or start a terminal to run the notebook on the local computer.
    - Navigate to the project folder or clone the repository to a desired location.
    - Export the corresponding notebooks to be executed as python script. 
2. Set `run_name` in `1_datasetGenerator.py` and `2_trainingLoop.py` to a desired name (must be the same in both) 
3. run bash commands:
    ```bash
    # start tmux session
    tmux

    # run the scripts and pipe the output to a log.txt file.
    python 1_datasetGenerator.py > log.txt
    python 2_trainingLoop.py > log.txt
    ```
4. Ctrl+B, then D (to detach the tmux session)
5. Open tensorboard with the ./runs folder

-> Reconnect to detached Session:
- `tmux ls`to list sessions:
    ```bash
    0: 1 windows (created Mon Mar 18 21:36:24 2024)
    1: 1 windows (created Wed Apr  3 10:22:08 2024)
    ```
- Open Session with: `tmux attach-session -t NUMBER`
    ```bash
    # to resume the example session 1 created on Wed Apr 3, attach session 1:
    tmux attach-session -t 1
    ```


### Optuna Dashboard Setup
In order to visualize the results of hyperparameter tuning, these can be made visible with the Optuna Dashboard. 
More information about the dashboard and getting started can be found here: https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html

> [!NOTE]
> The dashboard can also be installed and used without the `environment.yml` dependencies if only the results from the hyperparameter tuning of the thesis are to be analyzed. 

1. Install the dashboard and the recommended dependencies to speed up the dashboard:
    ```bash
    pip install optuna-dashboard

    pip install optuna-fast-fanova gunicorn
    ```
2. Open Optuna results in Dashboard:
    ```bash
    optuna-dashboard sqlite:///optuna/A6000_OptunaRun_2048.db
    ```
> [!NOTE]
> The database path for the file is optuna/A6000_OptunaRun_2048. If you want to analyze your own results, change the command to your own path: 
> `optuna-dashboard sqlite:///{YOUR PATH}`
3. Open the dashboard in your browser using the provided information from the optuna-dashboard command (For example: `Listening at: http://127.0.0.1:8080`). 


### Log in to Huggingface Hub
1. Test if huggingface-cli is installed and if already logged in.
    ```bash
    huggingface-cli whoami
    ```
2. Install huggingface-cli (Skip this, if already installed).
    ```bash
    pip install -U "huggingface_hub[cli]"
    ```
3. Login using the huggingface-cli (Skip this if you are already logged in).
    ```bash
    huggingface-cli login
    ```
    After this you should be prompted to past an Access Token. Generate the Access Token with the required rights through the Hugging Face website and your user account.
3. Gain access to the model repository if required. To do this, log in to the Hugging Face Hub, go to the model repository and agree to the corresponding Terms of Services or similar. 


> [!NOTE] 
> For more information on the huggingface-cli see the documentation: [Command Line Interface (CLI)](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)

## Contributing
This project is not being actively developed further, for questions or suggestions please open an issue.

## Acknowledgements
- **Filip Stepniak** ([feelsteps](https://github.com/feelsteps)) - Supervisor from [esentri](https://esentri.com/)
- **Mario Herb** ([chuckson](https://github.com/chuckson))- Supervisor from [esentri](https://esentri.com/)
- **Prof. Patrick Baier** ([pabair](https://github.com/pabair)) - Supervising professor from [Hochschule Karlsruhe - University of Applied Sciences](https://www.h-ka.de/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.