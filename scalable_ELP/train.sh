python train.py --init_channels 64 --pres_channels 64 ## you can modify channel number 64 into the 512, which
# is the original number the paper. %512 can help you get the better result as those in paper
--log_dir '/home/chengshuai/data/ELPunfolding/result/scaleble_ELP_a40'
--trainset_dir '/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/JPEGImages/480/'
--valset_dir '/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-test/JPEGImages/480p/'
--valset_cha_dir '/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-test-challenge/JPEGImages/480p/'
--dataset_split_files '/home/chengshuai/data/ELPunfolding/data/traindata/datasetname/'
--code_dir '/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/code'
--orig_dir '/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/orig3'
--order_dir '/home/chengshuai/data/ELPunfolding/data/traindata/DAVIS-480-train/order_color'
## You can get more parameters in train.py to modify.             