import torch
from data import LyricsCommentsDatasetPsuedo_fusion
from torch import utils, nn
from model import CommentGenerator
from model_fusion import CommentGenerator_fusion
import transformers
import datasets
from tqdm import tqdm
import statistics
import os
DATASET_PATH = "/home/jespina6/CSE575/CSE575-T37Project/data/dataset_test.pkl"
MODEL_PATH = "/home/jespina6/Downloads/bart_fusion_full.pt"
# MODEL_NAME = "bart"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_dataset = LyricsCommentsDatasetPsuedo_fusion(DATASET_PATH)
dataset_length = len(test_dataset)

test_dataloader = utils.data.DataLoader(test_dataset,
                                         # batch_size=len(valid_dataset),
                                         batch_size=32,
                                         shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'baseline' in MODEL_PATH:
    #model = CommentGenerator().cuda()
    model = CommentGenerator().to(device)
else:
    #model = CommentGenerator_fusion().cuda()
    model = CommentGenerator_fusion().to(device)
model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

samples_list = list()
# generate
for batch_index, [lyrics, comment, music_id] in enumerate(tqdm(test_dataloader)):
    if 'baseline' in MODEL_PATH:
        with torch.no_grad():
            output_samples = model.generate(lyrics)
    else:
        with torch.no_grad():

            print(music_id)
            output_samples = model.generate(lyrics, music_id)
    samples_list.append(output_samples)

# ------ ROUGE ------ #

metrics = datasets.load_metric('rouge')#, 'sacrebleu', 'meteor', 'bertscore')

for batch_index, [lyrics, comment, music_id] in enumerate(tqdm(test_dataloader)):
    output_samples = samples_list[batch_index]
    metrics.add_batch(predictions=output_samples, references=comment)

score = metrics.compute()
print(score)

# ------ BLEU ------ #

metrics = datasets.load_metric('sacrebleu')#, 'sacrebleu', 'meteor', 'bertscore')

for batch_index, [lyrics, comment, music_id] in enumerate(tqdm(test_dataloader)):
    output_samples = samples_list[batch_index]
    metrics.add_batch(predictions=output_samples, references=[[i] for i in comment])

score = metrics.compute()
print(score)

# ------ BERTScore ------ #

metrics = datasets.load_metric('bertscore')#, 'sacrebleu', 'meteor', 'bertscore')

for batch_index, [lyrics, comment, music_id] in enumerate(tqdm(test_dataloader)):
    output_samples = samples_list[batch_index]
    metrics.add_batch(predictions=output_samples, references=[[i] for i in comment])

score = metrics.compute(lang='en')
score = statistics.mean(score['f1'])
print(score)

# ------ METEOR ------ #

metrics = datasets.load_metric('meteor')#, 'sacrebleu', 'meteor', 'bertscore')

for batch_index, [lyrics, comment, music_id] in enumerate(tqdm(test_dataloader)):
    output_samples = samples_list[batch_index]
    metrics.add_batch(predictions=output_samples, references=[[i] for i in comment])

score = metrics.compute()
print(score) 
