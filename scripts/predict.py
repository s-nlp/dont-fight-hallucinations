import argparse
from pathlib import Path

import jsonlines
import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from model.utils import get_custom_scores, aggregation


def predict(facts, nli_model, ent_w, cont_w, neutral_w, thr=0):
    nli_scores = get_custom_scores(facts, nli_model)
    agg_score = aggregation(*nli_scores, ent_w=ent_w, cont_w=cont_w, neutral_w=neutral_w)
    return int(agg_score < thr)

def main(args):
    nli_model = CrossEncoder(args.nli_model_name, device=args.device)
    results = []

    with jsonlines.open(args.facts_file, mode="r") as reader:
        for obj in tqdm(reader):
            ex = {}
            for name, facts in obj.items():
                ex[name] = predict(facts, nli_model, args.ent_w, args.cont_w, args.neutral_w)
            results.append(ex)

    with jsonlines.open(args.prediction_file, mode="w") as writer:
        for res in results:
            writer.write(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli_model_name", type=str, default="cross-encoder/nli-deberta-v3-large")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--facts_file", type=str, default="facts.jsonl")
    parser.add_argument("--ent_w", type=float, default=1.75)
    parser.add_argument("--cont_w", type=float, default=-2)
    parser.add_argument("--neutral_w", type=str, default=0)
    parser.add_argument("--prediction_file", type=str, default="prediction.jsonl")
    args = parser.parse_args()

    main(args)
