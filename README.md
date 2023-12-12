# llm-pe

# setup

## install dependencies
```
pip install -r requirements.txt
```


## setup API keys
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


