import torch
import torch.nn.functional as F


class Example:
    def __init__(self, expand_dict, init_seq, tokens=[], seq=None, score=0):
        self.expand_dict = expand_dict
        self.init_seq = init_seq
        self.score = score
        self.seq = init_seq if seq is None else seq
        self.tokens = tokens

    def extend(self, model):
        expand_set = self.expand_dict[tuple(self.tokens)]
        new_expand_set = []
        if len(expand_set) > 0:
            with torch.no_grad():
                expand_set_idx = torch.tensor(expand_set, dtype=torch.long).cuda()
                input_ids = torch.tensor(self.seq, dtype=torch.long).cuda().unsqueeze(0)
                sequence_output = model.roberta(input_ids, return_dict=False)[0]
                prediction_scores = model.lm_head(sequence_output[0, -1])
                prediction_scores = -F.log_softmax(prediction_scores, dim=-1)[expand_set_idx]

            for idx, w in enumerate(expand_set):
                new_expand_set.append(Example(
                    self.expand_dict,
                    self.init_seq,
                    self.tokens + [w],
                    self.seq + [w],
                    self.score + prediction_scores[idx].item(),
                ))
        return new_expand_set


def cons_beam_search(init_seq, entities, expand_dict, model, k=50):
    expand_set = [Example(expand_dict, init_seq)]
    all_sequences = []
    while len(expand_set) > 0:
        new_expand_set = []
        for example in expand_set:
            new_expand_set += example.extend(model)
        all_sequences += new_expand_set
        new_expand_set = [example for example in new_expand_set if len(expand_dict[tuple(example.tokens)]) > 0]
        expand_set = sorted(new_expand_set, key=lambda x: x.score)[:k]
    results = []
    entities = set(entities)
    for example in all_sequences:
        predicted = tuple(example.tokens)
        if predicted in entities:
            results.append((predicted, example.score))
    results = sorted(results, key=lambda x: x[1])
    results = [r[0] for r in results]
    return results
