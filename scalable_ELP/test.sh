python test.py --init_channels 64 --pres_channels 64 ## you can modify channel number 64 into the 512, which
# is the original number the paper. %512 can help you get the better result as those in paper
--log_dir '/home/chengshuai/data/ELPunfolding/result/scaleble_ELP_a40'
--code_dir '/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/code_2050_25'
--orig_dir '/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/origd24n'
## You can get more parameters in test.py to modify.    
             