# Prix-LM

Code for paper [Prix-LM: Pretraining for Multilingual Knowledge Base Construction](https://arxiv.org/abs/2110.08443).

## Requirements
* [PyTorch](http://pytorch.org/)
* [Transformers](https://github.com/huggingface/transformers)
* wandb
* tqdm

## Training and Evaluation

We have released the code for link prediction. Code on other tasks will be released soon.

### Link Prediction
Before running link prediction, unzip the evaluation data under the ``data/link_prediction`` folder. Run link prediction on our DBpedia test set by:

```bash
>> python lp.py --lan language
```

The ``lan`` parameter can take the 9 lanugages in our paper (en, it, fr, de, fi, et, hu, tr, and ja).


This project is supported by by the National Science Foundation of United States Grant IIS 2105329.

```
@inproceedings{zhou-etal-2022-prix,
    title = "Prix-{LM}: Pretraining for Multilingual Knowledge Base Construction",
    author = "Zhou, Wenxuan  and
      Liu, Fangyu  and
      Vuli{\'c}, Ivan  and
      Collier, Nigel  and
      Chen, Muhao",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.371",
    pages = "5412--5424"
}

```
