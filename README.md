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
