from training_pipeline import stage1_prepare_data
from training_pipeline import stage2_prepare_data



if __name__=="__main__":
    stage1_prepare_data.preparedata(config_path="config.yaml")
    stage2_prepare_data.featurization(config_path="config.yaml")
    

    