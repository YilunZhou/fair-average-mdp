
import pickle, os
from tqdm import trange, tqdm
from tqdm.contrib import tzip

import numpy as np
from scipy.optimize import linprog
import cvxpy as cp
import matplotlib.pyplot as plt

class SMD():
	def __init__(self, Gamma, r, rho, I_hat, M, eta_x, eta_lmd):
		self.Gamma = np.array(Gamma)
		self.r = np.array(r)
		self.rho = rho
		self.I_hat = np.array(I_hat)
		self.M = M
		self.num_s = self.Gamma.shape[0]
		self.num_sa = len(self.r)
		self.num_a = self.num_sa // self.num_s
		self.eta_x = eta_x
		self.eta_lmd = eta_lmd
		self.solved = False

	def grad_x(self, x, lmd):
		sa = np.random.choice(range(self.num_sa))
		s = sa // self.num_a
		s_next = np.random.choice(range(self.num_s), p=self.Gamma[:, sa])
		grad_x = np.zeros(self.num_sa)
		grad_x[sa] = self.num_sa * (lmd[s_next] - lmd[s] - self.r[sa])
		return grad_x

	def grad_lmd(self, x, lmd):
		x[x < 0] = 0
		x = x / x.sum()
		sa = np.random.choice(range(self.num_sa), p=x)
		s = sa // self.num_a
		s_next = np.random.choice(range(self.num_s), p=self.Gamma[:, sa])
		grad_lmd = np.zeros(self.num_s)
		grad_lmd[s_next] = - 1
		grad_lmd[s] = grad_lmd[s] + 1
		return grad_lmd

	def argmax_x(self, x_t, lmd_t):
		grad_x_t = self.grad_x(x_t, lmd_t)
		x = cp.Variable(self.num_sa)
		constraints = [self.I_hat @ x >= self.rho, cp.sum(x) == 1, x >= 0]
		x_t[x_t < 0] = 0
		objective = cp.Minimize(
						cp.sum(cp.multiply(grad_x_t * self.eta_x, x)) - \
						cp.sum(cp.multiply(x_t, cp.log(x)))
					)
		prob = cp.Problem(objective, constraints)
		result = prob.solve()
		return x.value

	def argmin_lmd(self, x_t, lmd_t):
		grad_lmd_t = self.grad_lmd(x_t, lmd_t)
		lmd = cp.Variable(self.num_s)
		constraints = [-2 * self.M <= lmd, lmd <= 2 * self.M]
		objective = cp.Minimize(
						cp.sum(cp.multiply(grad_lmd_t * self.eta_lmd, lmd)) + \
						cp.norm(lmd - lmd_t) ** 2)
		prob = cp.Problem(objective, constraints)
		result = prob.solve()
		return lmd.value

	def solve(self, n_iter, x_init=None, lmd_init=None):
		assert not self.solved, 'the SMD has been solved'
		self.solved = True
		if x_init is None:
			x_init = np.ones(self.num_sa) / self.num_sa
		if lmd_init is None:
			lmd_init = np.zeros(self.num_s)
		x = np.array(x_init)
		lmd = np.array(lmd_init)
		self.xs = [x]
		self.lmds = [lmd]
		for t in trange(n_iter, ncols=70):
			x_new = self.argmax_x(x, lmd)
			lmd_new = self.argmin_lmd(x, lmd)
			self.xs.append(x_new)
			self.lmds.append(lmd_new)
			x = x_new
			lmd = lmd_new
		return self.xs, self.lmds

	def get_transition_matrix(self, pi):
		T = np.zeros((self.num_s, self.num_s))
		for i in range(self.num_s):
			for j in range(self.num_s):
				T[j, i] = sum([pi[i, a] * self.Gamma[j, i * self.num_a + a] 
							   for a in range(self.num_a)])
		return T

	def get_policy(self, x):
		policy = np.array(x).reshape(self.num_s, self.num_a)
		policy = policy / policy.sum(axis=1).reshape(-1, 1)
		return policy

	def get_stationary_distribution(self, x):
		policy = self.get_policy(x)
		T = self.get_transition_matrix(policy)
		eig_val, eig_vec = np.linalg.eig(T)
		phi = np.real(eig_vec[:, 0])
		phi = phi / phi.sum()
		return phi

	def get_expected_reward(self, x):
		policy = self.get_policy(x)
		phi = self.get_stationary_distribution(policy)
		sa_dist = policy * phi.reshape(-1, 1)
		return (sa_dist.flatten() * self.r).sum()


def LP(Gamma, r, rho, I_hat):
	num_s = Gamma.shape[0]
	num_a = Gamma.shape[1] // num_s
	I_all = np.ones(len(r)).reshape(1, -1)
	res = linprog(c=-r, A_eq=I_hat-Gamma, b_eq=[0] * num_s,
					    A_ub=np.vstack((I_all, -I_hat)), b_ub=np.concatenate(([1], -rho)))
	avg_R = -res.fun
	return avg_R

def gap(smd, x, lmd):
	'''
	gap = max_x' f(x', lmd) - min_lmd' f(x, lmd')
	'''
	def f(x_, lmd_):
		return np.dot(smd.r, x_) + np.dot(np.dot(lmd_, smd.I_hat - smd.Gamma), x_)
	G = smd.I_hat - smd.Gamma
	def max_x():
		x_var = cp.Variable(smd.num_sa)
		constraints = [smd.I_hat @ x_var >= smd.rho, cp.sum(x_var) == 1, x_var >= 0]
		objective = cp.Maximize(cp.sum(cp.multiply(x_var, smd.r)) + 
								cp.sum(cp.multiply(lmd, (G) @ x_var))
							   )
		prob = cp.Problem(objective, constraints)
		result = prob.solve()
		return result
	def min_lmd():
		lmd_var = cp.Variable(smd.num_s)
		constraints = [-2 * 100 <= lmd_var, lmd_var <= 2 * 100]
		objective = cp.Minimize(np.dot(x, smd.r) + 
								cp.sum(cp.multiply(lmd_var, np.dot(x, G.T)))
							   )
		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver='CVXOPT')
		return result
	gap = max_x() - min_lmd()
	return gap


def convergence(T=20000, N=100):
	r = [1, 0.1, 0.1, 0.1, 0.1, 0.1]
	r = np.array(r)

	rho = [0.1, 0.1, 0.25]
	rho = np.array(rho)

	I_hat = [[1, 1, 0, 0, 0, 0],
			 [0, 0, 1, 1, 0, 0],
			 [0, 0, 0, 0, 1, 1]]
	I_hat = np.array(I_hat)

	Gamma = [[0.0, 0.0, 0.1, 0.9, 0.9, 0.1],
			 [0.9, 0.1, 0.0, 0.0, 0.1, 0.9],
			 [0.1, 0.9, 0.9, 0.1, 0.0, 0.0]]
	Gamma = np.array(Gamma)

	M = 100
	eta_x = 0.01
	eta_lmd = 0.01

	os.makedirs('saved_runs/convergence/', exist_ok=True)
	for i in range(0, N):
		smd = SMD(Gamma, r, rho, I_hat, M, eta_x, eta_lmd)
		smd.solve(T)
		pickle.dump(smd, open(f'saved_runs/convergence/{i}.pkl', 'wb'))

def different_rho(T=10000, N=10):
	r = [1, 0.1, 0.1, 0.1, 0.1, 0.1]
	r = np.array(r)

	I_hat = [[1, 1, 0, 0, 0, 0],
			 [0, 0, 1, 1, 0, 0],
			 [0, 0, 0, 0, 1, 1]]
	I_hat = np.array(I_hat)

	Gamma = [[0.0, 0.0, 0.1, 0.9, 0.9, 0.1],
			 [0.9, 0.1, 0.0, 0.0, 0.1, 0.9],
			 [0.1, 0.9, 0.9, 0.1, 0.0, 0.0]]
	Gamma = np.array(Gamma)

	M = 100
	eta_x = 0.01
	eta_lmd = 0.01

	os.makedirs('saved_runs/different_rho/', exist_ok=True)
	for rho2 in [0.10, 0.15, 0.20, 0.25, 0.30]:
		rho = [0.1, 0.1, rho2]
		rho = np.array(rho)
		for i in range(0, N):
			smd = SMD(Gamma, r, rho, I_hat, M, eta_x, eta_lmd)
			smd.solve(T)
			pickle.dump(smd, open(f'saved_runs/different_rho/{rho2:0.2f}_{i}.pkl', 'wb'))

def plot_convergence(N=100):
	real_rss = []
	fake_rss = []
	state_freqss = []
	for run in range(N):
		smd = pickle.load(open(f'saved_runs/convergence/{run}.pkl', 'rb'))
		xs = smd.xs[1:]
		xs_cumsum = np.cumsum(xs, axis=0)
		T = len(xs)
		xs_avg = xs_cumsum / np.array(range(1, T + 1)).reshape(-1, 1)
		real_rs = []
		fake_rs = []
		state_freqs = []
		for t in trange(T, ncols=70):
			x_use = xs_avg[t]
			fake_rs.append(np.dot(smd.r, x_use))
			real_rs.append(smd.get_expected_reward(x_use))
			state_freqs.append(smd.get_stationary_distribution(x_use))
		fake_rss.append(fake_rs)
		real_rss.append(real_rs)
		state_freqss.append(state_freqs)
	optimal_R = LP(smd.Gamma, smd.r, smd.rho, smd.I_hat)

	avg_real_rs = np.mean(real_rss, axis=0)
	std_real_rs = np.std(real_rss, axis=0)
	avg_fake_rs = np.mean(fake_rss, axis=0)
	std_fake_rs = np.std(fake_rss, axis=0)
	plt.figure(figsize=[4, 3])
	plt.plot(range(1, T + 1), avg_real_rs)
	plt.fill_between(range(1, T + 1), avg_real_rs - std_real_rs, avg_real_rs + std_real_rs, 
					 color='C0', alpha=0.3)
	plt.plot(range(1, T + 1), avg_fake_rs)
	plt.fill_between(range(1, T + 1), avg_fake_rs - std_fake_rs, avg_fake_rs + std_fake_rs, 
					 color='C1', alpha=0.3)
	plt.plot([1, T], [optimal_R, optimal_R])
	plt.xlabel('T')
	plt.ylabel('Reward')
	plt.legend(['$r^T x$', '$r^T x^\\epsilon$', 'Optimal Reward'])
	plt.savefig('convergence_avg_reward.pdf', bbox_inches='tight')

	plt.figure(figsize=[4, 3])
	avg_state_freqs = np.mean(state_freqss, axis=0)
	std_state_freqs = np.std(state_freqss, axis=0)
	for j in range(avg_state_freqs.shape[1]):
		plt.plot(avg_state_freqs[:, j])
		plt.fill_between(range(1, T + 1), avg_state_freqs[:, j] - std_state_freqs[:, j], 
						 avg_state_freqs[:, j] + std_state_freqs[:, j], 
						 color=f'C{j}', alpha=0.3)
	plt.legend([f'State {j}' for j in range(avg_state_freqs.shape[1])])
	plt.plot([1, T], [0.25, 0.25], 'C2--')
	plt.xlabel('T')
	plt.ylabel('State Frequency')
	plt.savefig('convergence_state_freq.pdf', bbox_inches='tight')

def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:,1] / (extrema[:,1] - extrema[:,0])
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]
    tot_span = tops[1] + 1 - tops[0]

    extrema[0,1] = extrema[0,0] + tot_span * (extrema[0,1] - extrema[0,0])
    extrema[1,0] = extrema[1,1] + tot_span * (extrema[1,0] - extrema[1,1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]

def plot_gap(N=100):
	gapss = []
	for i in range(N):
		print('iter', i)
		smd = pickle.load(open(f'saved_runs/convergence/{i}.pkl', 'rb'))
		xs = smd.xs[1:]
		lmds = smd.lmds[1:]
		xs_avg = np.cumsum(xs, axis=0) / np.arange(1, len(xs) + 1).reshape(-1, 1)
		lmds_avg = np.cumsum(lmds, axis=0) / np.arange(1, len(lmds) + 1).reshape(-1, 1)
		Ts = np.linspace(0, len(xs), 1001).astype('int')
		Ts[-1] -= 1
		gaps = []
		for t in tqdm(Ts, ncols=70):
			gaps.append(gap(smd, xs_avg[t], lmds_avg[t]))
		gapss.append(gaps)
	avg = np.mean(gapss, axis=0)
	std = np.std(gapss, axis=0)

	real_rss = []
	fake_rss = []
	state_freqss = []
	for run in range(N):
		smd = pickle.load(open(f'saved_runs/convergence/{run}.pkl', 'rb'))
		xs = smd.xs[1:]
		xs_cumsum = np.cumsum(xs, axis=0)
		T = len(xs)
		xs_avg = xs_cumsum / np.array(range(1, T + 1)).reshape(-1, 1)
		real_rs = []
		fake_rs = []
		state_freqs = []
		for t in tqdm(Ts, ncols=70):
			x_use = xs_avg[t]
			fake_rs.append(np.dot(smd.r, x_use))
			real_rs.append(smd.get_expected_reward(x_use))
			state_freqs.append(smd.get_stationary_distribution(x_use))
		fake_rss.append(fake_rs)
		real_rss.append(real_rs)
		state_freqss.append(state_freqs)
	optimal_R = LP(smd.Gamma, smd.r, smd.rho, smd.I_hat)
	diff_rss = optimal_R - np.array(real_rss)
	avg_diff = np.mean(diff_rss, axis=0)
	std_diff = np.std(diff_rss, axis=0)

	plt.figure(figsize=[5, 3])
	ax1 = plt.gca()
	ax1.plot(Ts, avg)
	ax1.fill_between(Ts, avg - std, avg + std, color='C0', alpha=0.3)
	ax1.tick_params(axis='y', labelcolor='C0')
	ax1.set_ylabel('$Gap(x, \\lambda)$', color='C0')
	plt.xlabel('T')
	ax2 = plt.gca().twinx()  # instantiate a second axes that shares the same x-axis
	ax2.plot(Ts, avg_diff, 'C1')
	ax2.fill_between(Ts, avg_diff - std_diff, avg_diff + std_diff, color='C1', alpha=0.3)
	ax2.tick_params(axis='y', labelcolor='C1')
	ax2.set_ylabel('$r^T x^{*} - r^T x$', color='C1')
	align_yaxis(ax1, ax2)
	plt.tight_layout()
	plt.savefig('convergence_gap.pdf', bbox_inches='tight')

def plot_rho(N=10):
	avg_rs = []
	std_rs = []
	avg_state_freqs = []
	std_state_freqs = []
	rhos = [0.10, 0.15, 0.20, 0.25, 0.30]
	optimal_Rs = []
	for rho in rhos:
		real_rss = []
		state_freqss = []
		for run in range(N):
			smd = pickle.load(open(f'saved_runs/different_rho/{rho:0.2f}_{run}.pkl', 'rb'))
			xs = smd.xs[1:]
			xs_cumsum = np.cumsum(xs, axis=0)
			T = len(xs)
			xs_avg = xs_cumsum / np.array(range(1, T + 1)).reshape(-1, 1)
			real_rs = []
			state_freqs = []
			for t in trange(T, ncols=70):
				x_use = xs_avg[t]
				real_rs.append(smd.get_expected_reward(x_use))
				state_freqs.append(smd.get_stationary_distribution(x_use))
			real_rss.append(real_rs)
			state_freqss.append(state_freqs)
		avg_rs.append(np.mean(real_rss, axis=0))
		std_rs.append(np.std(real_rss, axis=0))
		avg_state_freqs.append(np.mean(state_freqss, axis=0))
		std_state_freqs.append(np.std(state_freqss, axis=0))
		optimal_Rs.append(LP(smd.Gamma, smd.r, smd.rho, smd.I_hat))

	T = len(avg_rs[0])
	plt.figure(figsize=[4, 3])
	xs = range(1, T + 1)
	for i, (avg, std) in enumerate(zip(avg_rs, std_rs)):
		plt.plot(xs, avg)
		plt.fill_between(xs, avg - std, avg + std, alpha=0.3)
		plt.plot([xs[0], xs[-1]], [optimal_Rs[i], optimal_Rs[i]], f'C{i}--')
	plt.savefig('rhos_reward.pdf', bbox_inches='tight')
	plt.figure(figsize=[4, 3])
	for i, (avg, std) in enumerate(zip(avg_state_freqs, std_state_freqs)):
		plt.plot(xs, avg[:, 2], f'C{i}')
		plt.fill_between(xs, avg[:, 2] - std[:, 2], avg[:, 2] + std[:, 2], 
						 color=f'C{i}', alpha=0.3)
		plt.plot([xs[0], xs[-1]], [rhos[i], rhos[i]], f'C{i}--')
	plt.savefig('rhos_state_freq.pdf', bbox_inches='tight')

if __name__ == '__main__':
	convergence(T=20000, N=100)
	plot_convergence(N=100)
	plot_gap(N=100)

	different_rho(T=10000, N=10)
	plot_rho(N=10)
