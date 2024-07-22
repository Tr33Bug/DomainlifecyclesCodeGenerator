# DomainlifecyclesCodeGenerator
The *Domainlifecycles Code Generator (DCG)* formerly *NitroX Code Generator (NCG)* is the first version of a Generative AI Assistance System which was developed for esentri's Domain Lifecycles Framework for the Domain-Driven Design (DDD) Development process.

This first version of the DCG is able to create syntactically correct Domainlifecycles JSON objects as part of the Domainlifecycles DSL. More information about the DCG and its creation as well as limitations and future work can be found in the [Master Thesis PDF](Master-Thesis_Götz-Henrik_Wiegand_2024.pdf). 

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
├── images
│   └── ColoredDataPreprocessingProcess.jpg
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
├── 0_1_datasetGenerator.ipynb
├── DCG_Server.py
└── README.md
```

### Folder Structure
- **all_json:** 
    - Folder containing the raw JSON files for fine-tuning the DCG. 
    - The customer-related project data from the "esentri-Partner" was removed from the data set. That was 80% of the files and so the data set is only stored here as an example.
- **datasets:**
    - Empty folder reserved for export and storage of the generated and cleaned dataset.
- **images:**
    - Folder for diagrams and graphics for project documentation.
- **models:**
    - Reserved path for the model export with the `finalTraining_v1` model as the result of the final training for the DCG. 
- **runs:**
    - Reserved tensorboard callback folder for the training history logs. The results from the final training of the DCG are stored here. 
    - The metrics and progressions logged there can be displayed and analyzed with a tensorboard.

## Installation and Setup



## Contributing
This project is not being actively developed further, for questions or suggestions please open an issue.

## Acknowledgements
- **Filip Stepniak** ([feelsteps](https://github.com/feelsteps)) - Supervisor from [esentri](https://esentri.com/)
- **Mario Herb** ([chuckson](https://github.com/chuckson))- Supervisor from [esentri](https://esentri.com/)
- **Prof. Patrick Baier** ([pabair](https://github.com/pabair)) - Supervising professor from [Hochschule Karlsruhe - University of Applied Sciences](https://www.h-ka.de/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.