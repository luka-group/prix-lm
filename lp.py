import argparse
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from os.path import join
from decode import cons_beam_search
from tqdm import tqdm
from collections import defaultdict
import wandb

expand_dict = defaultdict(list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/link_prediction", type=str)
    parser.add_argument("--model_name_or_path", default="wzhouad/prix-lm", type=str)
    parser.add_argument("--lan", default="et", type=str)
    parser.add_argument("--k", default=50, type=int)
    args = parser.parse_args()
    wandb.init(project="LP", name="{}_{}".format(args.lan, args.k))

    triples = []
    entities = []
    entity_dict = defaultdict(list)
    with open(join(args.data_dir, 'triples.txt'), 'r') as fh:
        for line in fh:
            e1, r, e2, lan = line.strip().split('\t')
            if lan == args.lan:
                triples.append((e1, r, e2, lan))

    with open(join(args.data_dir, 'entities.txt'), 'r') as fh:
        for line in fh:
            ent, lan = line.strip().split('\t')
            if lan == args.lan:
                entities.append(ent)

    with open(join(args.data_dir, 'filtered.txt'), 'r') as fh:
        for line in fh:
            e1, r, e2, lan = line.strip().split('\t')
            if lan == args.lan:
                entity_dict[(e1, r)].append(e2)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    model.to(0)

    cls = [tokenizer.cls_token_id]
    sep = [tokenizer.sep_token_id]
    e1_mark = tokenizer.convert_tokens_to_ids(['[S]'])
    r_mark = tokenizer.convert_tokens_to_ids(['[P]'])
    e2_mark = tokenizer.convert_tokens_to_ids(['[O]'])
    eos_mark = tokenizer.convert_tokens_to_ids(['[EOS]'])

    def tokenize(x):
        x = tokenizer.tokenize(x)
        x = tokenizer.convert_tokens_to_ids(x)
        return x

    token_ents = [tuple(tokenize(ent) + eos_mark) for ent in entities]
    for ent in token_ents:
        for i in range(len(ent)):
            expand_dict[ent[:i]].append(ent[i])
    for key, value in expand_dict.items():
        expand_dict[key] = list(set(value))

    c1, c10, c3, ca = 0.0, 0.0, 0.0, 0.0
    for step, t in enumerate(tqdm(triples)):
        e1, r, e2 = t[:3]
        gold = tuple(tokenize(e2) + eos_mark)
        init_seq = cls + e1_mark + tokenize(e1) + sep + sep + r_mark + tokenize(r) + sep + sep + e2_mark

        t_ents = []
        for idx, en in enumerate(entities):
            if (en == e2 or en not in entity_dict[(e1, r)]):
                t_ents.append(token_ents[idx])

        model.eval()
        results = cons_beam_search(init_seq, t_ents, expand_dict, model, k=args.k)
        ca += 1.0
        if results[0] == gold:
            c1 += 1.0
        if gold in results[:10]:
            c10 += 1.0
        if gold in results[:3]:
            c3 += 1.0
        hits1 = c1 / ca
        hits10 = c10 / ca
        hits3 = c3 / ca
        wandb.log({'hits1': hits1, 'hits10': hits10, 'hits3': hits3}, step=step)


if __name__ == "__main__":
    main()
