feature_extraction.py
    Used for generating data for pretraining, the code gets the data needed to train Normwear. The commented out code contains code for generating data for handcrafted features.

train_dqn_warmstart_pretrain.py
    Used to pretrain dqn model using data generated from pretrained Normwear rolling out on the environment

train_dqn.py
    Used to train RL for event start or end detection using stable-baselines 3

train_WESAD_supervised_NormWear.py
    Used for NormWear finetuning on WESAD. Note that opt=NormWear  is the only one that works right now, and it trains Normwear based on data from WESAD_Data_6s_6s.parquet. 
    For completion sake, here's the other 2 opt that was supposed to be supported, they don't work right now. Getting them to work require minimal changes though.
        opt=NormWear_Embed is meant to train MLP on embedding from NormWear pretrained by the original author of the paper.
        opt=NormWear_CWT is meant to test averaging the CWT over 10 timesteps instead of averaging 10 signal segments. This is very slow and probably doesn't work right now.

train_WESAD_supervised.py
    Used for training on handcrafted features on WESAD. Only opt=Handcraft works, nothing else does

test_dqn_WESAD.py
    Used to visualize result of RL training

CWT_save.py(not important)
    Used to generate the averaged CWT of signals and save to pandas dataframe. I added it here just for completion sake.
