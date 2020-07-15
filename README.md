# A Multi-Layer and Multi-Ensemble Stock Trader Using Deep Learning and Deep Reinforcement Learning



## Abstract 

> Abstract The use of computer-aided stock trading is gaining popularity in recent years, mainly because of its ability to process efficiently past information through machine learning in order to predict future market behavior. Several approaches have been proposed this task, with the most effective ones using fusion of a pile of classifiers decisions to predict future stock values. However, using prices information only has proven to lead to poor results, mainly because market history is not enough to be an indicative of future market behavior. In this paper, we propose to tackle this issue by proposing a multi-layer and multi-ensemble stock trader. Our method starts by pre-processing data with hundreds of deep neural networks. Then, a reward-based classifier acts as a meta-learner to maximize profit and generate stock signals through different iterations. Finally, several metalearner trading decisions are fused in order to get a more robust trading. Experimental results of index Futures intra-day trading indicate better performance when compared to several other ensemble techniques and the conventional Buy-and-Hold strategy.


## Authors: 

- Salvatore Carta
- Andrea Corriga
- Anselmo Ferreira
- Alessandro Sebastian Podda
- Diego Reforgiato Recupero

# Info 
This is the source code of the paper "A Multi-Layer and Multi-Ensembled Stock Trader Using Deep Learning and Deep Reinforcement Learning"

In this source code, we offer two datasets from 2 SP500 symbols: JPM and MSFT (please check ./datasets folder). The source code runs JPM, to change for another one, just change DeepQTrading.py and ensemble.py according.

To execute the code, just run ./run_all_experiments.sh in your terminal, and 100 experiments will be done (check section 5.1 of our paper).

After running each experiment, a final pdf will be built with data from RL training and also a final table, showing the final trading results.

Please check source_code_map.png for a better explanation of what each .py file does.

If using our code, dont forget to cite our paper. 
