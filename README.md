# Fairness in Average MDP

This is the code repository for the AAMAS 2022 paper [_Long-Term Resource Allocation Fairness in Average Markov Decision Process (AMDP) Environment_](https://arxiv.org/abs/2102.07120) by Ganesh Ghalme, Vineet Nair, Vishakha Patil and Yilun Zhou. 

The code to reproduce the experiment results are in `smd.py`. Specifically, running
```bash
python smd.py
```
would automatically produce all the figures in the paper. 

The code should run with reasonably modern versions of `numpy`, `scipy`, `matplotlib`, `cvxpy`, and `tqdm`. But if you encounter any compatibility issues, below list the exact versions of these libraries used in the experiment. 
```
cvxopt==1.2.5.post1
cvxpy==1.1.7
ecos==2.0.7.post1
matplotlib==3.3.2
numpy==1.19.1
scipy==1.5.2
tqdm==4.56.0
```

Depending on the computation power of the computer, it may take more than ten hours to finish everything. The values of the total number of gradient descent steps `T` and the number of runs `N` can be decreased to reduce the computation time. However, `convergence()`, `plot_convergence()`, and `plot_gap()` use the same `N` value, and `different_rho()` and `plot_rho()` also use the same `N` value. 

The paper can be cited as
```
@inproceedings{ghalme2022long,
    title = {Long-Term Resource Allocation Fairness in Average Markov Decision Process (AMDP) Environment},
    author = {Ghalme, Ganesh, and Nair, Vineet and Patil, Vishakha and Zhou, Yilun},
    booktitle = {Proceedings of the International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS)},
    year = {2022},
    month = {May}
}
```
