# Methodology for Efficient CNN Architectures in Profiling Attacks
The current repository is associated with the article "<a href='...'>Methodology for efficient CNN architectures in Profiling Attacks</a>" available on the <a href='https://eprint.iacr.org/'>eprints</a>


Each dataset is composed of the following scripts and repositories:
- <b>cnn_architecture.py</b>: provides the script in order to train the model introduced in the article,
- <b>exploit_pred.py</b>: computes the evolution of the right key and saves the resulted picture,
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


## Citation

If you use our code, models or wish to refer to our results, please use the following BibTex entry:
```
@misc{cryptoeprint:2019:803,
    author = {Gabriel Zaid and Lilian Bossuet and Amaury Habrard and Alexandre Venelli},
    title = {Methodology for Efficient CNN Architectures in Profiling Attacks},
    howpublished = {Cryptology ePrint Archive, Report 2019/803},
    year = {2019},
    note = {\url{https://eprint.iacr.org/2019/803}},
}
```
