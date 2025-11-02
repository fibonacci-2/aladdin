README

#Datasets:
##ClusterlabAi/101_billion_arabic_words_dataset
113G	data/arabic_101B
30B

##pain/Arabic-Tweets
23G	data/Arabic-Tweets
6B tokens



#Training Runs
##2.1: 2.1_20251030_044938
Arabic-Twitter, Free Transformer, 
found 59 shards for split train
=> calculated gradient accumulation steps: 8
total_batch_size = 524288 
B = 128 # micro batch size
T = 512 # sequence length
max_steps = 11444  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

##2.2: 2.2_20251030_044938
2.1 with arabic 101B