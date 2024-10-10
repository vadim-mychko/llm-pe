# llm-pe

# Setup

## Install Dependencies
```
pip install -r requirements.txt
```


## Setup API Keys
create a `.env` file in the root directory with the following contents:

```
OPENAI_API_KEY=<your key>
```
and run
```
dotenv run python experiment_manager.py -config <path_to_config_file>
```


#pytrec-eval install note:
If pytrec-eval does not install, use `pytrec-eval-terrier``. If the installation does not work, you may try:

```
pip install --upgrade --no-cache-dir --use-deprecated=legacy-resolver pytrec-eval-terrier==0.5.1
```

If you get an error about "Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/" follow the installation in that link and try in.

# Usage

## Example workflow

### Creating a config file
We use config files to specify the parameters for an experiment. To start, you will want to modify one of our base config files (in the config folder) to use as a base config file from which your actual experiment config files will be derived. 

### Creating an experiment folder
Once you have a base config file, you can use Parameter Writer.ipynb to create the config files for your experiment.

### Running your experiment
To run your experiment, use the format

```
python ./experiment_manager.py -exp_dir=[path/to/experiment/folder]
```

In each experiment subfolder, this will create a results.json with the experiment details and a results folder with the per-user results. The per-user results allows for checkpointing in case of interruptions during a run.

### Evaluating your experiment
To evaluate your experiment results, use the format 
```
python ./eval_manager.py -exp_dir=[path/to/experiment/folder]
```

This will create a file in your experiment folder called aggregated_results.csv. This contains the evaluation data.

### Plotting your results
This section is less automated and will require more work from the user. The plotting functions used are in david_plots_annotated.ipynb. Execution details are in that folder.

You may find it useful at this point to combine your results from various aggregated results files into a straightforward file structure that allows for indexing by dataset, method, etc. I created a file structure in the organized_results folder. I used tools/eval_all.ipynb to automate needing to copy everything over each time. This relies on eval_map.json, which specifies the experiment directory to organized_results directory mapping.

