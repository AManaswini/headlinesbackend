"""Spins up a user input that gives you a headline, and asks you to "rephrase" it.

In the process, it will provide multiple scores for you to evaluate and continually reproduce your headline.
"""

# create a loading bar so user can see progress, then begin imports
import tqdm
import os

# def clear_terminal() -> None:
#     _ = os.system("cls") if os.name == "nt" else os.system("clear")

# clear_terminal()

# because it takes ten seconds to initialize, start by telling user their instructions and then begin loading
# models
# output_instructions_long = """\
# You will be given a headline from the online news platform 'Upworthy'. Your task is to
# re-write the headline to produce better user engagement. We will measure your performance
# on this task using several statistical models of user engagement, that predict how well
# your headline would perform. However, your re-written headline must be a reasonable 
# alternative to the original headline, ie. while you can adjust the syntax and structure of
# the headline, you can not change the meaning or interpretation. You may have as many attempts 
# as you like. Simply type your new headline and press 'Enter'. The input is case-insensitive, 
# so you do not need to worry about capitalizations.

# Here are the scores we will evaluate your headline on:
#   * MCTR-R: a simple model of user engagement, trained on the Upworthy dataset. High scores 
#     are better. Scores generally range from -0.02 to 0.02. High scores are better.
#   * MCTR-LM: a more complex model of user engagement, trained on the Upworthy dataset. High 
#     scores are better. Similar range to MCTR-R.
#   * Human: a model trained to detect human-written headlines. High scores are better: 1 means 
#     the model believes the headline was written by a human, 0 means the model believes the
#     headline was written by a computer.
#   * Consistent: a model trained to detect headlines that are consistent with the topic 
#     of the article. High scores are better: 1 means the model believes the proposed headline 
#     is consistent with the article, 0 means the model believes the proposed headline is 
#     inconsistent.
#   * 100 Chars: a model trained to detect headlines that are less than 100 characters. 1 if
#     the headline is less than 100 characters (good), 0 otherwise.

# You may also use the following additional commands:
#   * help (or simply enter a blank line): see these instructions again
#   * next: try again with a new headline
#   * quit (or control-C): ends the program
#   * [not implemented] suggest: get the computer to suggest some alternatives for you
# """

# print(output_instructions_long)
# print("\nWe are just starting up the scoring interface, please wait ten seconds...")

progress_init = tqdm.tqdm(total=10, desc="Loading".ljust(16), leave=False, bar_format="{l_bar}{bar:50}")
progress_init.update(1)
progress_init.update(1)  # seems to fix bug by updating twice?

# now begin with imports while user reads instructions
import copy
import functools
import pathlib
import random
import sys
import time
import typing

# update bar here so user doesn't get impatient
progress_init.update(1)

import joblib
import pandas
import sentence_transformers
import sklearn
import titlecase
import torch
import transformers

# step progress for user
progress_init.update(1)

# load morphing tools
sys.path.append("experiments/x08_morphing")
import s01_morphtools as morphtools
from s34_aggregate_model import score_aggregate

progress_init.update(1)


## GLOBAL CONSTANTS

# just a buunch of model values
BATCH_SIZE = 32
MODEL_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = pathlib.Path("data/temp")


## DATA IMPORT
# get headlines & outcomes
headline_raw_df = pandas.read_feather(pathlib.Path("data/clean/headlines.feather"))
headline_ctr_df = pandas.read_feather(pathlib.Path("data/clean/headline_ctr.feather"))
progress_init.update(1)

# load CTR-LM model
model_state_dict = torch.load(
    OUTPUT_DIR.joinpath("model-ctr_prediction-state_dict-best.pt"), map_location=MODEL_DEVICE
)
model_ctrlm = transformers.AutoModelForSequenceClassification.from_pretrained(
    "bert-large-cased", num_labels=1, cache_dir=morphtools.MODEL_CACHEDIR, state_dict=model_state_dict
)
model_ctrlm_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-large-cased")
model_ctrlm_transform = torch.nn.Identity()
model_ctrlm.to(MODEL_DEVICE)
model_ctrlm.eval()
progress_init.update(1)

# load CTR-ridge model
model_sbert_ridge = joblib.load(OUTPUT_DIR.joinpath("morphing-embedding_model.pkl"))
model_sbert_emb = sentence_transformers.SentenceTransformer(
    "sentence-transformers/stsb-roberta-large", cache_folder=morphtools.MODEL_CACHEDIR
)
model_sbert_emb.to(MODEL_DEVICE)
model_sbert_emb.eval()
progress_init.update(1)

# get penalty-discriminator model
model_penalty_disc_state_dict = torch.load(
    OUTPUT_DIR.joinpath("model-discriminator-state_dict-best.pt"), map_location=MODEL_DEVICE
)
model_penalty_disc = transformers.AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=1, cache_dir=morphtools.MODEL_CACHEDIR, state_dict=model_penalty_disc_state_dict
)
model_penalty_disc_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
model_penalty_disc_transform = torch.nn.Sigmoid()
model_penalty_disc.to(MODEL_DEVICE)
model_penalty_disc.eval()

# penalty-topic model
model_penalty_topic_state_dict = torch.load(
    OUTPUT_DIR.joinpath("model-topic_penalty-state_dict-best.pt"), map_location=MODEL_DEVICE
)
model_penalty_topic = transformers.BertForNextSentencePrediction.from_pretrained(
    "bert-base-uncased", cache_dir=morphtools.MODEL_CACHEDIR, state_dict=model_penalty_topic_state_dict
)
model_penalty_topic_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model_penalty_topic.to(MODEL_DEVICE)
model_penalty_topic.eval()

# update user again
progress_init.update(1)


## HELPER CLASSES

# scoring functions
def score_headlines_rdige(
    text: str,
    embedding_model: torch.nn.Module,
    scoring_model: sklearn.linear_model.RidgeCV,
    batch_size: int = 16,
    device: str = "cpu",
) -> pandas.Series:
    text_embeddings = embedding_model.encode([text], batch_size=batch_size, device=device)
    text_embeddings = pandas.DataFrame(text_embeddings)
    text_embeddings.columns = text_embeddings.columns.astype("str")
    text_scores_ridge = scoring_model.predict(text_embeddings)
    return text_scores_ridge


def score_headlines_ctrlm(
    text: str, tokenizer, device: torch.device, model: torch.nn.Module, transform: torch.nn.Module
):
    # get ctr-lm prediction
    text_enc = tokenizer(text, return_tensors="pt", padding=True).to(device)
    text_output_raw = model(**text_enc)
    text_scores_ctrlm = transform(text_output_raw["logits"].squeeze(1)).to("cpu")
    return text_scores_ctrlm


def score_headlines_discr(
    text: str, tokenizer, device: torch.device, model: torch.nn.Module, transform: torch.nn.Module
):
    text_enc = tokenizer(text, return_tensors="pt", padding=True).to(device)
    text_output_raw = model(**text_enc)
    text_scores_penaltydisc = transform(text_output_raw["logits"].unsqueeze(1)).to("cpu")
    return text_scores_penaltydisc


def score_headlines_topic(
        text_new: str, text_baseline: str, tokenizer, device: torch.device, model: torch.nn.Module
):
    text_enc = tokenizer([[text_new, text_baseline]], return_tensors="pt", padding=True).to(device)
    text_output_raw = model(**text_enc)
    text_score_penaltytopic = text_output_raw["logits"].softmax(1)[:, 1].to("cpu")
    return text_score_penaltytopic


def breakline(s: str, line_width: int = 50) -> str:
    """Breaks a string according to the given line_width, at spaces"""
    words = s.split(" ")
    line_curr = ""
    lines = []
    for word in words:
        if len(line_curr) + len(word) + 1 > line_width:
            lines.append(line_curr.strip())
            line_curr = word + " "
        else:
            line_curr += word + " "
    lines.append(line_curr.rstrip())
    return "\n".join(lines)


## DATA PREP
headline_df = (
    headline_raw_df[["trial_id", "headline_id", "headline", "trial_random_value", "train_test_set"]]
    .merge(headline_ctr_df[["trial_id", "headline_id", "ctr_delta_bayesmean_v1"]], on=["trial_id", "headline_id"])
    .dropna(subset=["headline"])
    .query("train_test_set == 'validation'")
)
progress_init.update(1)

## SCORING PARTIAL FUNCTIONS
score_ridge = functools.partial(
    score_headlines_rdige, embedding_model=model_sbert_emb, scoring_model=model_sbert_ridge, device=MODEL_DEVICE
)
score_ctrlm = functools.partial(
    score_headlines_ctrlm,
    tokenizer=model_ctrlm_tokenizer,
    model=model_ctrlm,
    device=MODEL_DEVICE,
    transform=model_ctrlm_transform,
)
score_discr = functools.partial(
    score_headlines_discr,
    tokenizer=model_penalty_disc_tokenizer,
    model=model_penalty_disc,
    device=MODEL_DEVICE,
    transform=model_penalty_disc_transform,
)

def score_length(text: str) -> float:
    return torch.tensor([1 if len(text) <= 100 else 0])


# scoring system for produced headline
# right now, this class is hard-coded to expect very specific kinds of models. Eventually we will want to replace
# it with a family of ambiguous "scorer" functions, and force the user to create the single-parameter models
# themselves?
class Scorer(object):
    def __init__(self, scorers: typing.Dict[str, typing.Callable], device: torch.device) -> None:
        self.scorers = scorers
        self.device = device
        self._history = []
        self._linewidth = 50  # impacts table printing only

    def score(self, text: str) -> typing.Dict[str, float]:
        scores = {k: f(text).item() for k, f in self.scorers.items()}
        self._history.append(copy.deepcopy(scores))
        self._history[-1]["text"] = text
        return scores

    @property
    def history(self) -> pandas.DataFrame:
        table = pandas.DataFrame.from_records(self._history)
        table = table[["text"] + list(table.columns[:-1])]
        return table

    def __str__(self) -> str:
        table = self.history
        table["text"] = table["text"].apply(lambda s: breakline(s, self._linewidth))
        return table.to_markdown(tablefmt="grid")




progress_init.update(1)
progress_init.close()



def getinput():
#get first sentence

    # begin by sampling a single ID
    # row_begin = headline_df.sample(1)["headline_id"].iloc[0]
    # # extract ID and headline from the sample
    # text_begin = row_begin["headline"]
    # id_begin = row_begin["headline_id"]
    # # see if printing works now
    # print(f'id:{id_begin}') # this prints out text, that's correct
    # print(f'text:{text_begin}')  # this prints out a whole dataframe row! Incorrect!
    # # (insert remaining lines)
    # begin by sampling a single ID
    row_begin = headline_df.sample(1)
    print(f'row_begin:{row_begin}')
    # extract ID and headline from the sample
    text_begin = row_begin["headline"].iloc[0]
    print(f'text_begin:{text_begin}')
    id_begin = row_begin["headline_id"].iloc[0]
    print(f'id:{id_begin}') # this prints out text, that's correct
    text_begin = text_begin.lower()
    text_begin = titlecase.titlecase(text_begin)
    text_curr = text_begin
    return (text_curr, id_begin)

def findscore(text_curr):   
    score_topic = functools.partial(
    score_headlines_topic,
    text_baseline=text_curr,
    tokenizer=model_penalty_topic_tokenizer,
    model=model_penalty_topic,
    device=MODEL_DEVICE
)
# "MCTR": score_aggregate
    scorer = Scorer({"MCTR-R": score_ridge, "MCTR-LM": score_ctrlm, "Human": score_discr, "Consistent": score_topic, "100 chars": score_length}, device=MODEL_DEVICE)
    text_scores = scorer.score(text_curr)
    # score_original = scorer.history["MCTR-R"].iloc[0]
    # score_last = text_scores["MCTR-R"]
    # score_best = scorer.history["MCTR-R"].iloc[:-1].max()
    # result = {'MCTR-P':scorer.history["MCTR-R"].iloc[0],'MCTR-LM':scorer.history["MCTR-LM"].iloc[0],'Human':scorer.history["Human"].iloc[0]
    #           ,'Consistent':scorer.history["Consistent"].iloc[0]}
    res=score_aggregate(text_curr).item()
    result= {'Headline':text_curr,'Score':res,'Consistent':scorer.history["Consistent"].iloc[0]}
    print(result)
    return result




