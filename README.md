# stack-overflow-project__

utils.py
This has 2 function
1.process_posts
to process the semistructured data to tsv format this is used
2.read_yaml
to read the custom yaml file that we have prepared which is used to read all the necessary directory names....

training_pipeline/stage1_preparedata.py This has 1 main function named prepare_data and one input called config_path which is used for giving path of config.yaml
This will process an entire data and store in artifacts/prepare_data directory

config.yaml
This consists of key and value pairs where all the metadata of directory names are available.

merge_pipeline.py
This is the final script where we merge all the particular training_pipeline modules and call at once...

inferencing pipeline
This will help us to load the trained models and transformer objects and help to inference with the new data that will be either batch data or either the single user input
