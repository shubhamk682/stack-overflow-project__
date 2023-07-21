import os,sys
sys.path.append(os.getcwd())
import utils1 as utils
STAGE="STAGE 01 PREPARE DATA"

def preparedata(config_path):
    config=utils.read_yaml(config_path=config_path)
    input_data=config["local_data_file_path"]
    split_ratio=float(config["split_ratio"])
    artifacts=config["artifacts"]
    prepared_data_dir=os.path.join(artifacts["ARTIFACTS_DIR"],artifacts["PREPARED_DATA"])
    os.makedirs(prepared_data_dir,exist_ok=True)
    train_data_path=os.path.join(prepared_data_dir,artifacts["TRAIN_DATA"])
    test_data_path=os.path.join(prepared_data_dir,artifacts["TEST_DATA"])
    with open(input_data,"r",encoding="utf-8") as f_in:
        with open(train_data_path,"w",encoding="utf-8") as train_data:
            with open(test_data_path,"w",encoding="utf-8") as test_data:
                utils.process_posts(f_in=f_in.read(),target_tag="<python>",f_out_train=train_data,f_out_test=test_data,split=split_ratio)
  

preparedata("config.yaml")








