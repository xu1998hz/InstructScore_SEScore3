# InstructScore (SEScore3)

An amazing explanation metric (diagnostic report) for text generation evaluation

First step, you may download all required dependencies through: pip3 install -r requirements.txt

Second step, by running following codes, weight can be directly downloaded from Huggingface or you can download weight from this Google Drive link: https://drive.google.com/drive/folders/1seBqoewWHgu7I_AmZ6FE-_3EcJ3mGWQ2?usp=sharing

<div  align="center"> 
<img src="figs/InstructScore_teaser.jpg" width=400px>
</div>

To run our metric, you only need five lines

````
```
from InstructScore import *
refs = ["Normally the administration office downstairs would call me when there’s a delivery."]
outs = ["Usually when there is takeaway, the management office downstairs will call."]
scorer = InstructScore()
batch_outputs, scores_ls = scorer.score(refs, outs)
```
````

![Overview](figs/InstructScore.jpg)


