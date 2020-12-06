# Methodology for Efficient CNN Architectures in Profiling Attacks
The current repository is associated with the article "<a href='https://tches.iacr.org/index.php/TCHES/article/view/8391'>Methodology for efficient CNN architectures in Profiling Attacks</a>" available on <a href='https://tches.iacr.org/index.php/TCHES/index'>IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)</a> and the <a href='https://eprint.iacr.org/'>eprints</a>


Each dataset is composed of the following scripts and repositories:
- <b>cnn_architecture.py</b>: provides the script in order to train the model introduced in the article,
- <b>exploit_pred.py</b>: computes the evolution of the right key and saves the resulted picture (<b>Credit</b>: Damien Robissout),
- <b>(Optionnal) clr.py</b>: computes the One-Cycle Policy (see "<a href='https://arxiv.org/abs/1708.07120'>Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates
</a>" and "<a href='https://arxiv.org/abs/1803.09820'>A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay</a>,
- <b>"training_history"</b>: contains information related to the loss function and the accuracy,
- <b>"model_predictions"</b>: contains information related to the model predictions,
- <b>"fig"</b>: contains the figure related to the rank evolution,
- <b>"..._trained_models"</b>: containts the model used in the article.

The trace sets were obtained from publicly databases: 
- <b>DPA-contest v4</b>: http://www.dpacontest.org/v4/42_traces.php
- <b>AES_HD dataset</b>: https://github.com/AESHD/AES_HD_Dataset
- <b>AES_RD dataset</b>: https://github.com/ikizhvatov/randomdelays-traces
- <b>ASCAD</b>: https://github.com/ANSSI-FR/ASCAD


## Raw data files hashes
The zip file SHA-256 hash value is:
<hr>

**AES_HD/AES_HD_dataset.zip:**
`00a3d02f01bae8c4fcefda33e3d1adb57bed0509ded3cdcf586e213b3d87e41b`

<hr>

**AES_RD/AES_RD_dataset/AES_RD_attack.zip:**
`379c0e29e7f2b7e24ca2ece40b83200b083d48afabd6eabbb01f8ed38a42ebcf`
**AES_RD/AES_RD_dataset/AES_RD_profiling.zip:**
`93a77b83df7e54656fce798c184e4fb4e3cdc5a740758c0432bdb8c7bd58154d`

<hr>

**ASCAD/N=0/ASCAD_dataset.zip:**
`5f5924e2d0beca5b57fbc48ace137dbb2fe12dd03976aa38f4a699ab21e966b0`
**ASCAD/N=50/ASCAD_dataset.zip:**
`9bf704727390a73cf67d3952bc2cacef532b0b62e55f85d615edaa6cd8521f51`
**ASCAD/N=100/ASCAD_dataset.zip:**
`2d803db27e58fec3d805cd3cf039b303cad1e0c9ea7a8102a07020bd07113cd9`

<hr>

**DPA-contest v4/DPAv4_dataset.zip:**
`c42e0626793848ad38634f1765354fbecd9df3fa606ceb593a94febe6ebeda1f`

<hr>

## Citation

If you use our code, models or wish to refer to our results, please use the following BibTex entry:
```
@article{Zaid_Bossuet_Habrard_Venelli_2019, 
title={Methodology for Efficient CNN Architectures in Profiling Attacks},
volume={2020},
url={https://tches.iacr.org/index.php/TCHES/article/view/8391}, 
DOI={10.13154/tches.v2020.i1.1-36}, 
number={1}, 
journal={IACR Transactions on Cryptographic Hardware and Embedded Systems}, 
author={Zaid, Gabriel and Bossuet, Lilian and Habrard, Amaury and Venelli, Alexandre}, 
year={2019}, 
month={Nov.}, 
pages={1-36} 
}
```
