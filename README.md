This is the source code of the paper "A Multi-Layer and Multi-Ensembled Stock Trader Using Deep Learning and Deep Reinforcement Learning"

Authors: Anselmo Ferreira, Alessandro Sebastian Podda and Andrea Corriga 

In this source code, we offer two datasets from 2 SP500 symbols: JPM and MSFT (please check ./datasets folder). The source code runs JPM, to change for another one, just change DeepQTrading.py and ensemble.py according.

To execute the code, just run ./run_all_experiments.sh in your terminal, and 100 experiments will be done (check section 5.1 of our paper).

After running each experiment, a final pdf will be built with data from RL training and also a final table, showing the final trading results.

Please check source_code_map.png for a better explanation of what each .py file does.

If using our code, dont forget to cite our paper. 
