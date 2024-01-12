import yaml
import json
import argparse
import os
import csv
from pathlib import Path
from utils.setup_logging import setup_logging
import dataloaders
import numpy as np
import pytrec_eval
import datetime
import pandas as pd
from collections import OrderedDict
import re
from scipy.stats import norm

class EvalManager:

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        config_path = os.path.join(exp_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                self.config = yaml.safe_load(config_file)
        else:
            raise FileNotFoundError(f"Could not find {config_path}")
        
        self.logger = setup_logging(self.__class__.__name__, self.config)

    def eval_experiments(self):
        """
        This function walks through each directory in the experiments directory specified in the config file. 
        For each experiment directory, it runs eval_experiment method to evaluate the results.

        It uses pytrec_eval library for parsing qrels file and evaluating the results. 
        The exact evaluation metrics are specified in the config file, these include per query metrics, and aggregated metrics for all queries

        Eval results are written: 
            - for each individual experiment: in the respective subdirectories in the experiments_directory
            - for the set of all experiments in experiments directory: in the root of experiments directory
            - for all experiments to date: in a master csv
        """

        # Get QREL ground truth 
        qrels_path = self.config['data']['user_path'] # TODO: Double check if missing users causes issues between experiments

        with open(qrels_path, "r") as qrels_file:
            self.qrels = pytrec_eval.parse_qrel(qrels_file)

        # Get metrics for evaluation
        metrics_config = self.config['metrics']['metrics']
        
        if isinstance(metrics_config, dict):
            metrics = metrics_config
        elif metrics_config == 'supported_measures':
            metrics = pytrec_eval.supported_measures
        else:
            raise ValueError(f"Invalid metrics configuration: {metrics_config}")
                   
        # Get metrics to be averaged across queries
        self.metrics_to_avg = self.config['metrics']['to_avg']

        evaluator = pytrec_eval.RelevanceEvaluator(
            self.qrels, metrics)
        
        self.experiments_dir = self.config['paths']['experiment_dir']

        # results will be written as rows of a csv
        all_rows = []

        # os.walk() will yield a tuple containing directory path, 
        # directory names and file names in the directory.
        for root, dirs, files in os.walk(self.experiments_dir):
            # iterate over directories only
            dirs.sort()
            for directory in dirs:
                exp_dir = os.path.join(root, directory)
                self.logger.debug(f'running evaluator on {exp_dir}')
                row = self.eval_experiment(evaluator, exp_dir)
                if row is not None:
                    all_rows.append(row)

        self.write_to_csv(all_rows)

        #TODO: MODIFY CODE TO WORK WITH PER_TURN FILES

    def eval_experiment(self, evaluator, exp_dir):
        """
        This function takes a evaluator and an experiment directory as input. It evaluates the results 
        of the experiment using the evaluator and writes the evaluation output to 'eval_results.json' 
        file in the experiment directory.

        Args:
            evaluator (pytrec_eval.RelevanceEvaluator): An instance of pytrec_eval.RelevanceEvaluator used for evaluating the results.
            exp_dir (str): The directory path where the experiment results are stored.

        """
        ci_results_list = []
        mean_eval_results_list = []
        for turn in range(self.config['dialogue_sim']['num_turns']):
            exp_results_path = os.path.join(exp_dir, f"trec_results_turn{turn}.txt")

            # Check if TREC file exists before processing
            if not os.path.isfile(exp_results_path):
                self.logger.error(f"Missing TREC file at {exp_results_path}. Skipping this experiment.")
            
            # TODO: Handle duplicates in TREC file (failure of LLM to understand no duplicates)
                
            with open(exp_results_path, "r") as results_file:
                results = pytrec_eval.parse_run(results_file)

            per_query_eval_results = evaluator.evaluate(results)

            output_file = os.path.join(exp_dir, 'per_query_eval_results_turn%d.json' % turn)

            # Write the per-query results to the output file
            with open(output_file, "w") as f:
                json.dump(per_query_eval_results, f, indent=4)

            # Get the mean results
            mean_eval_results = self.get_mean_eval_results(per_query_eval_results)

            # Get confidence intervals
            conf_lvl = float(self.config['ci']['confidence_level'])
            ci_results = self.get_ci_results(per_query_eval_results, conf_lvl) 

            ci_results_list.append(ci_results)
            mean_eval_results_list.append(mean_eval_results)

        row = self.get_row(mean_eval_results_list, ci_results_list, exp_dir)

        return row


    def get_ci_results(self,per_query_eval_results, conf_lvl):
        """
        Calculate confidence intervals for specified metrics.
    
        Parameters:
        - per_query_eval_results (dict): Dictionary of query results with metrics.
        - conf_lvl (float): Confidence level as a proportion (e.g., 0.95 for 95%).
    
        Returns:
        - dict: Lower and upper bounds of the confidence interval for each metric.
    
        Note:
        - Assumes a large sample size.
        """

        ci_results = {}
        
        for metric in self.metrics_to_avg:

            metric_vals = [query_results[metric] for query_results in per_query_eval_results.values() if metric in query_results]

            #get sample standard deviation
            ssd = np.std(metric_vals,ddof=1)

            #get z value for desired confidence level. 
            z = norm.ppf((1+conf_lvl)/2)

            #get margin of error
            me = z * ssd/np.sqrt(len(metric_vals))

            mean = np.mean(metric_vals)

            ci_results[f"{metric}_lb"] = mean - me
            ci_results[f"{metric}_ub"] = mean + me

        return ci_results
    
    def get_mean_eval_results(self,per_query_eval_results):

        """
        This function takes per_query_eval_results as inputs.
        It returns mean values for each metric in self.metrics_to_avg across all queries.
        """
        eval_results = {}
        for metric in self.metrics_to_avg:
            eval_results[metric] = np.mean([query_results[metric] for query_results in per_query_eval_results.values() if metric in query_results])
        return eval_results


    def write_to_csv(self, all_rows):
        """
        Writes all the rows into the 'aggregated_results.csv' file in the experiments directory 
        and appends them into a master csv file specified in the config.

        Args:
            all_rows (list of lists): Each inner list is a row that contains the experiment name, timestamp, 
            specific config values and evaluation results.
        """

        headers = ["Experiment Name", "Timestamp", "PE Module", "Response Update", "Num Recs", "Item Selection", "Item Scorer",
               "Preprocess Query", "Preprocessor Name", "Entailment Model", "LLM Temperature",
               "Data Path", "User Path"] 
        for turn_num in range(self.config['dialogue_sim']['num_turns']):
            metrics_w_turn = [[f"{metric}@{turn_num}", f"{metric}@{turn_num}_lb", f"{metric}@{turn_num}_ub"] for metric in list(self.metrics_to_avg)]
            for metric_set in metrics_w_turn:
                headers.extend(metric_set)

        # Write to aggregated_results.csv
        df = pd.DataFrame(all_rows, columns=headers)
        df.to_csv(os.path.join(self.experiments_dir, 'aggregated_results.csv'), index=False)

    
    def get_row(self, mean_eval_results_list, ci_results_list, exp_dir):
        """
        This function takes mean_eval_results and an experiment directory as input.
        It reads the config.yaml file from the experiment directory and combines
        the information from mean_eval_results and the config file into a row.
        """
        experiment_name = exp_dir.rsplit(',', 1)[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        with open(os.path.join(exp_dir, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        row = [
            experiment_name,
            timestamp,
            config['pe']['pe_module_name'],
            config['pe']['response_update'],
            config['pe']['num_recs'],
            config['query']['item_selection'],
            config['item_scoring']['item_scorer_name'],
            config['item_scoring']['preprocess_query'],
            config['item_scoring']['history_preprocessor_name'],
            config['llm']['entailement_model'],
            config['llm']['temperature'],
            config['data']['data_path'],
            config['data']['user_path']
        ]
        for turn_num in range(self.config['dialogue_sim']['num_turns']):
            row.extend(mean_eval_results_list[turn_num].values())
            row.extend(ci_results_list[turn_num].values())
        return row
    
    def json_to_trec_results(self, folder_path):
        # Load in json results data
        with open(folder_path + "/results.json") as json_data:
            results = json.load(json_data)
        # Convert to QREL format
        for turn_num in range(self.config['dialogue_sim']['num_turns']):
            qrel_rows = []
            for user_id, user_data in results.items():
                # Create an entry for each item recommended at this turn for this
                for item_rank, item_id in enumerate(user_data['rec_items'][turn_num]):
                    item_string = "%s Q0 %s %s %s standard\n" % (user_id, item_id, str(item_rank + 1), str(len(user_data['rec_items'][turn_num]) - item_rank))
                    qrel_rows.append(item_string)
            # Print to file
            with open(folder_path + f"/trec_results_turn{turn_num}.txt", "w") as out_file:
                for row in qrel_rows:
                    out_file.write(row)

    def convert_trecs_in_dir(self):
        # os.walk() will yield a tuple containing directory path, 
        # directory names and file names in the directory.
        for root, dirs, files in os.walk(self.exp_dir):
            # we are interested in directories only
            for directory in dirs:
                folder_path = os.path.join(root, directory)
                self.json_to_trec_results(folder_path)

def run_eval_on_dir(exp_dir):
    em = EvalManager(exp_dir)
    em.convert_trecs_in_dir()
    em.eval_experiments()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-exp_dir", "--experiment_dir", type=str)

    args = parser.parse_args()

    run_eval_on_dir(args.experiment_dir)
    #run_eval_on_dir("./experiments/anton_dt_methods_jan_9_pt7")
