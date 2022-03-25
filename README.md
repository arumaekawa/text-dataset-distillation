# Text Dataset Distillation

for deep learning study 2022

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/arumaekawa/text-dataset-distillation.git
   ```
2. Install packages:
   ```
   pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```
3. In `src/env`, set the value of DATA_ROOT_DIR as data directory and MODEL_ROOT_DIR as model directory

## Examples

Note: All command execute in `src` directory

test example of distilled data (created by my experiment)

```
./examples/run_distilled_data.sh --random_init --label_type hard --n_inner_steps 1 --pretrained_distilled_data ../distilled_data_examples/distilbert_ag_news_1_random_init_hard_inner_step_1
```

## Options

## Source files

```
src
├── all_dataset_attrs.json
├── distill.py
├── env
├── evaluate.py
├── examples
│   ├── run_distilled_data.sh
│   ├── run_full_data.sh
│   └── run_random_data.sh
├── full_data.py
├── main.py
├── model.py
├── random_data.py
├── requirements.txt
├── run.sh
├── settings.py
├── transformers_models
│   ├── __init__.py
│   └── modeling_distilbert.py
└── utils.py

3 directories, 20 files

```

## References

- Dataset Distillation [[paper]](https://arxiv.org/abs/1811.10959) [[code]](https://github.com/SsnL/dataset-distillation)
- Soft-Label Dataset Distillation and Text Dataset Distillation [[paper]](https://ieeexplore.ieee.org/document/9533769) [[code]](https://github.com/ilia10000/dataset-distillation)
- Dataset Distillation for Text Classification [[paper]](https://arxiv.org/abs/2104.08448)
-
