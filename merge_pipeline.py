from training_pipeline import stage1_prepare_data
from training_pipeline import stage2_prepared_data
from training_pipeline import stage3_training_data
from training_pipeline import stage4_evaluation



if __name__=="__main__":
    stage1_prepare_data.preparedata(config_path="config.yaml")
    stage2_prepared_data.featurization(config_path="config.yaml")
    stage3_training_data.training(config_path="config.yaml")
    stage4_evaluation.evaluate(config_path="config.yaml")
    

    