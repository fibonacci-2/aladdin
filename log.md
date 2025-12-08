=== PROCESSING STATISTICS ===
r_SaudiForSaudis: 1457098 raw, 1048058 cleaned
r_LGBTEgypt: 139416 raw, 97355 cleaned
r_arabs: 1092025 raw, 832689 cleaned
r_ExEgypt: 1544378 raw, 1218999 cleaned
r_iraq: 358096 raw, 250782 cleaned
r_egyptology: 21936 raw, 18659 cleaned
r_arabic: 79867 raw, 68885 cleaned
r_Egyptiandoctors: 59908 raw, 47938 cleaned
r_uae: 919780 raw, 755604 cleaned
r_saudiarabia: 2628842 raw, 1917075 cleaned
r_algeria: 1593429 raw, 1293438 cleaned
r_lebanon: 2520650 raw, 2001424 cleaned
r_Egypt: 2359569 raw, 1715406 cleaned
r_morocco: 2648655 raw, 2134811 cleaned
r_palestine: 1557648 raw, 1161750 cleaned
r_jordan: 1697526 raw, 1256457 cleaned
r_tunisia: 1439126 raw, 1150238 cleaned
r_CAIRO: 1407793 raw, 1087669 cleaned
r_askegypt: 255173 raw, 199970 cleaned
r_syria: 580771 raw, 444646 cleaned

Total posts/comments processed: 18701853

README

# Datasets:
##ClusterlabAi/101_billion_arabic_words_dataset
113G	data/arabic_101B
30B

## pain/Arabic-Tweets
23G	data/Arabic-Tweets
6B tokens

## Fineweb2-MSA
1B rows omartifical-intelligence-space/fineweb2-msa

## ASAD
path = kagglehub.dataset_download("asalhi/train-files")

# Training Runs
##2.1: 2.1_20251030_044938
Arabic-Twitter, Free Transformer, 
found 59 shards for split train
=> calculated gradient accumulation steps: 8
total_batch_size = 524288 
B = 128 # micro batch size
T = 512 # sequence length
max_steps = 11444  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

## 2.2: 2.2_20251030_044938
2.1 with arabic 101B


## 2.3: 
data has been combined from both datasets, one with matching the small dataset (119 shards) and 
another matching bigger dataset (601 shards). training for same # of steps on both.
data/combined_119 

got intereputed first run 2.3.1 and gave a decent model after 7k steps: model_05000_steps_5.5577_loss

## 2.4:
2.3 with data/combined_601

## 2.5:
fieweb2-msa //giving .29 mmlu at 5000 steps, compared to .25 with 101B at same steps!! 101B data is trash compared to this. confirmed by prompting the model on a qa example. even though it achieved higher val loss at this step (2.8).

## 2.6:
    fineweb2 
    
    block_size: int = 512
    vocab_size: int = 64000
    n_layer: int = 22          # 22 layers (as in 1.98B model)
    n_head: int = 64           # 64 attention heads (as in 1.98B model)
    n_embd: int = 3072  
    H: int = 8
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1

## 2.6.2
2.6 with proper medium model dimentions as per gpt2 medium config: as opposed to mhGPT random shit.
    block_size: int = 512
    vocab_size: int = 64000
    n_layer: int = 24          
    n_head: int = 16           
    n_embd: int = 1024  
    H: int = 8
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1

## 2.6.3
large flavor as per gpt2 configs
     block_size: int = 512
    vocab_size: int = 64000
    n_layer: int = 36          
    n_head: int = 20           
    n_embd: int = 1280  
    H: int = 8
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1

# finetuning
depression:
Checkpoint                                               Train_Acc Train_Recall Train_F1 Train_Precision Val_Acc Val_Recall Val_F1 Val_Precision Test_Acc Test_Recall Test_F1 Test_Precision
2.2_20251030_085832/log/model_05000_steps_2.8212_loss     57.50        0.780    0.646           0.551   57.81      0.796  0.649         0.548    56.28       0.737   0.624          0.540
adhd:
2.2_20251030_085832/log/model_05000_steps_2.8212_loss.pt     55.94        0.963    0.690           0.538   57.19      0.965  0.708         0.559    53.31       0.915   0.658          0.514
Epoch 1 Summary:
Training   - Acc: 56.83%, Recall: 0.377, F1: 0.468, Precision: 0.618
Validation - Acc: 56.81%, Recall: 0.371, F1: 0.457, Precision: 0.596