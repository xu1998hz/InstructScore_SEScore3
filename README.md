# InstructScore (SEScore3)

An amazing explanation metric (diagnostic report) for text generation evaluation

<div  align="center"> 
<img src="figs/InstructScore_teaser.jpg" width=400px>
</div>

## Installation
We list all required dependencies in requirements.txt. You can create a conda environment and install all dependencies through following commands (Higher python version is recommended):

```bash
    conda create -n instructscore python=3.12
    conda activate instructscore
    pip install -r requirements.txt
```

## Usage
There are two ways to use InstructScore.

### Option 1: Have a fast try through Huggingface
We have uploaded our model to Huggingface, which can be found [here](https://huggingface.co/xu1998hz/InstructScore).
You can directly try InstructScore via several lines of code:

```python
from InstructScore import InstructScore
# You can choose from 'mt_zh-en', 'caption', 'd2t', 'commonsense' or "key-to-text" to reproduce results in the paper
task_type = 'mt_zh-en' 
# Example input for X-English translation
refs = ["Normally the administration office downstairs would call me when there’s a delivery."]
outs = ["Usually when there is takeaway, the management office downstairs will call."]

# Example input for captioning generation
# task_type="caption"
# refs = ["The two girls are playing on a yellow and red jungle gym."]
# outs = ["The woman wearing a red bow walks past a bicycle."]

# Example input for table-to-text generation
# task_type="d2t"
# srcs = ["['Piotr_Hallmann | height | 175.26', 'Piotr_Hallmann | weight | 70.308']"]
# refs = ["Piotr Hallmann is 175.26 cm tall and weighs 70.308 kg."]
# outs = ["Piotr Hallmann has a height of 175.26 m and weights 70.308."]

# Example input for Commonsense text generation
# task_type="commonsense"
# srcs = ["food, eat, chair, sit"]
# refs = ["A man sitting on a chair eating food."]
# outs = ["a man eats food and eat chair sit in the beach."]

# Example input for keyword-to-text generation
# task_type="key-to-text"
# srcs = ["['X | type | placetoeat', "X | area | 'X'", 'X | pricerange | moderate', 'X | eattype | restaurant']"]
# refs = ["May I suggest the X? It is a moderately priced restaurant near X."]
# outs = ["X is a restaurant in X with a moderately priced menu."]

# Example input for English-to-German translation (Beta testing)
# task_type="mt_en-de"
# refs=["Warnung vor stürmischem Wetter, da starke Winde eine 'Lebensgefahr' darstellen"]
# outs=["Warnung vor stürmischem Wetter, da starke Winde Lebensgefahr darstellen können"]

# Example input for English-to-Russian translation (Beta testing)
# task_type="mt_en-ru"
# refs=["Нет, вы не сможете ввести дату встречи, вам нужно будет разместить заказ, и тогда мы сможем отложить предметы для вас, мы можем отложить их сначала на три месяца"]
# outs=["Нет, вы не сможете указать дату встречи, вам нужно будет оформить заказ, после чего мы сможем временно <v>приостановить производство</v> товаров для вас. Вначале мы можем отложить их на три месяца"]

# Example input for English-to-Spanish translation (Beta testing)
# task_type="mt_en-es"
# refs=["Y hay una distinción muy importante allí que veremos."]
# outs=["Y hay una distinción muy anormal allí que falta veremos."]

scorer = InstructScore(device_id=device_id, task_type=task_type, batch_size=6, cache_dir=None)
if task_type=="commonsense" or task_type=="d2t" or task_type == "key-to-text":
    batch_outputs, scores_ls = scorer.score(ref_ls=refs, out_ls=outs, src_ls=srcs)
else:
    batch_outputs, scores_ls = scorer.score(ref_ls=refs, out_ls=outs)
```


### Option 2: Download weight from Google Drive

You can also download the checkpoint from this Google Drive [link](https://drive.google.com/drive/folders/1seBqoewWHgu7I_AmZ6FE-_3EcJ3mGWQ2?usp=sharing).

### Reproduce main table results

```
cd reproduce
python3 process_result_bagel.py # process_result_{task}.py
```

### To train your own InstructScore

```
# Training code
deepspeed --num_gpus 8 code/finetune_llama.py --f <Your Instruction training data> --output_dir <Your saved weight dir> --max_length <Max length> --num_epoch <Epoch>
# You can use localhost to specify specific GPU
deepspeed --include localhost:1 code/finetune_llama.py --f <Your Instruction training data> --output_dir <Your saved weight dir> --max_length <Max length> --num_epoch <Epoch>
```

![Overview](figs/instructscore_main.png)

```bash
@inproceedings{xu-etal-2023-sescore2,
    title = "{SESCORE}2: Learning Text Generation Evaluation via Synthesizing Realistic Mistakes",
    author = "Xu, Wenda  and
      Qian, Xian  and
      Wang, Mingxuan  and
      Li, Lei  and
      Wang, William Yang",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.283/",
    doi = "10.18653/v1/2023.acl-long.283",
    pages = "5166--5183",
    abstract = "Is it possible to train a general metric for evaluating text generation quality without human-annotated ratings? Existing learned metrics either perform unsatisfactory across text generation tasks or require human ratings for training on specific tasks. In this paper, we propose SEScore2, a self-supervised approach for training a model-based metric for text generation evaluation. The key concept is to synthesize realistic model mistakes by perturbing sentences retrieved from a corpus. We evaluate SEScore2 and previous methods on four text generation tasks across three languages. SEScore2 outperforms all prior unsupervised metrics on four text generation evaluation benchmarks, with an average Kendall improvement of 0.158. Surprisingly, SEScore2 even outperforms the supervised BLEURT and COMET on multiple text generation tasks."
}

@inproceedings{xu-etal-2022-errors,
    title = "Not All Errors are Equal: Learning Text Generation Metrics using Stratified Error Synthesis",
    author = "Xu, Wenda  and
      Tuan, Yi-Lin  and
      Lu, Yujie  and
      Saxon, Michael  and
      Li, Lei  and
      Wang, William Yang",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.489/",
    doi = "10.18653/v1/2022.findings-emnlp.489",
    pages = "6559--6574",
    abstract = "Is it possible to build a general and automatic natural language generation (NLG) evaluation metric? Existing learned metrics either perform unsatisfactorily or are restricted to tasks where large human rating data is already available. We introduce SESCORE, a model-based metric that is highly correlated with human judgements without requiring human annotation, by utilizing a novel, iterative error synthesis and severity scoring pipeline. This pipeline applies a series of plausible errors to raw text and assigns severity labels by simulating human judgements with entailment. We evaluate SESCORE against existing metrics by comparing how their scores correlate with human ratings. SESCORE outperforms all prior unsupervised metrics on multiple diverse NLG tasks including machine translation, image captioning, and WebNLG text generation. For WMT 20/21En-De and Zh-En, SESCORE improve the average Kendall correlation with human judgement from 0.154 to 0.195. SESCORE even achieves comparable performance to the best supervised metric COMET, despite receiving no human annotated training data."
}


@inproceedings{xu-etal-2023-instructscore,
    title = "{INSTRUCTSCORE}: Towards Explainable Text Generation Evaluation with Automatic Feedback",
    author = "Xu, Wenda  and
      Wang, Danqing  and
      Pan, Liangming  and
      Song, Zhenqiao  and
      Freitag, Markus  and
      Wang, William  and
      Li, Lei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.365",
    doi = "10.18653/v1/2023.emnlp-main.365",
    pages = "5967--5994"
}
```
