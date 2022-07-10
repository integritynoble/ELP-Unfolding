"""

"""
import os
import glob
import torch
from numpy import savez_compressed
from numpy import load
from torch.utils.data.dataset import Dataset
from utils import open_sequence,get_patch,open_sequence_train

NUMFRXSEQ_VAL = 5	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

NUMFRXSEQ_TRAIN = 5	# number of frames of each sequence to include in TRAINidation dataset
TRAINSEQPATT = '*' # pattern for name of validation sequence

class TrainDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, train_files_names, trainsetdir=None,valsetdir=None,valsetdircha=None, orderdir=None, patchsize=96,epoach_num=0, gray_mode=False, present_set=None,video_length=20, num_input_frames=NUMFRXSEQ_TRAIN):
		self.gray_mode = gray_mode
		self.patch_size=patchsize
		self.video_length=video_length
		split_file = train_files_names + 'train.txt'		
		f = open(split_file,'r')
		TRAINSEQPATT = f.readlines()
		f.close()		
		# Look for subdirs with individual sequences
		
		seqs_dirs = []
		for i in range(len(TRAINSEQPATT)):
			file = os.path.join(trainsetdir, TRAINSEQPATT[i][:-1])
			seqs_dirs.append(file)

		split_file = train_files_names + 'val.txt'
		f = open(split_file,'r')
		TRAINSEQPATT = f.readlines()
		f.close()		
		# Look for subdirs with individual sequences
		
		for i in range(len(TRAINSEQPATT)):
			file = os.path.join(valsetdir, TRAINSEQPATT[i][:-1])
			seqs_dirs.append(file)

		split_file = train_files_names + 'val_cha.txt'
		f = open(split_file,'r')
		TRAINSEQPATT = f.readlines()
		f.close()		
		# Look for subdirs with individual sequences
		
		for i in range(len(TRAINSEQPATT)):
			file = os.path.join(valsetdircha, TRAINSEQPATT[i][:-1])
			seqs_dirs.append(file)


		# open individual sequences and append them to the sequence list
		seqs_dirs.sort()
		# seqs_dirs = seqs_dirs[:8]
		
		choose_num=epoach_num%16
		#count_num=double(count_num)
		if choose_num==0:
			count_num=epoach_num
			sequences = []
			for seq_dir in seqs_dirs:
				seq,count_num, _, _ = open_sequence_train(seq_dir, gray_mode,orderdir,count_num,choose_num,patchsize, expand_if_needed=False, \
								max_num_fr=num_input_frames)
				# seq is [num_frames, C, H, W]
				for ii in range(len(seq)):
					sequences.append(seq[ii])			
		else:
			sequences=present_set.sequences[0]
		
		sequences1=sequences[choose_num::16]#26000 25840 25900 26010 
		list=[]
		list.append(sequences)
		list.append(sequences1)
		self.sequences = list

	def __getitem__(self, index):
		samples = torch.from_numpy(self.sequences[1][index].copy())
		#samples =self.sequences[index]
		#samples = get_patch(samples, self.patch_size, self.video_length)
		return samples

	def __len__(self):
		return len(self.sequences[1])

