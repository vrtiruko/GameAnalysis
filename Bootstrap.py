#! /usr/bin/env python2.7

import RandomGames as RG

from GameIO import read, to_JSON_str, io_parser
from Regret import regret
from Nash import mixed_nash, replicator_dynamics, pure_nash
from RoleSymmetricGame import SampleGame, Profile, Game
from RandomGames import add_noise

from sys import stdin
from random import sample
from copy import copy, deepcopy
from functools import partial

from itertools import product, combinations_with_replacement as CwR, repeat,izip
import numpy as np
import json

def subsample(game, num_samples):
	"""
	Returns a game with a random subset of the input game's payoff samples.

	Note: this is not intended for use on games with different numbers of 
	samples per profile.
	"""
	sg = copy(game)
	sg.sample_values = map(lambda p: p[:,:,sample(range(p.shape[-1]), \
			num_samples)], sg.sample_values)
	sg.reset()
	sg.max_samples = num_samples
	return sg


def holdout(game, num_samples):
	"""
	Returns the same game as subsample(), but also the game consisting of the 
	remaining samples. Won't work if payoffs have different numbers of samples.
	"""
	game.makeArrays()
	sg = copy(game)
	withheld = copy(game)
	sample_values = deepcopy(sg.sample_values)
	sample_values = sample_values.swapaxes(0,3)
	np.random.shuffle(sample_values)
	sample_values = sample_values.swapaxes(0,3)
	sg.sample_values = sample_values[:,:,:,:num_samples]
	sg.max_samples = sg.min_samples = num_samples
	sg.reset()
	withheld.sample_values = sample_values[:,:,:,num_samples:]
	withheld.max_samples = withheld.min_samples = game.max_samples - num_samples
	withheld.reset()
	return sg, withheld


def pre_aggregate(game, count):
	"""
	Aggregates samples to produce a game with |samples| / count samples.

	Note: this is not intended for use on games with different numbers of 
	samples per profile.
	"""
	agg = copy(game)
	sv = agg.sample_values
	sv = np.swapaxes(sv, 0, -1)
	np.random.shuffle(sv)
	sv = np.swapaxes(sv, 0, -1)
	shape = list(sv.shape)
	samples = shape[-1]
	if samples % count != 0:
		sv = sv[:,:,:,:samples - (samples % count)]
	shape.append(count)
	shape[-2] /= count
	sv = np.reshape(sv, shape)
	sv = np.average(sv, axis=-1)
	agg.sample_values = sv
	agg.reset()
	agg.min_samples = agg.max_samples = shape[-2]
	return agg


def bootstrap(game, equilibria, intervals = [95], statistic=regret, method="resample", method_args=[], points=1000):
	"""
	Returns a bootstrap distribution for the statistic.

	To run a resample regret boostrap:
		bootstrap(game, eq)
	To run a single-sample regret bootstrap: 
		bootstrap(game, eq, method='single_sample')
	To run a replicator dynamics bootstrap:
		bootstrap(game, eq, replicator_dynamics)
	"""
	
	print "--Bootstrap Values--"
	
	boot_lists = [[] for x in xrange(0,len(equilibria))]
	combinations = []
	method = getattr(game, method)
	for __ in range(points):
		method(*method_args)
		combinations.append(np.array(game.values))
		count = 0
		for eq in equilibria:
			boot_lists[count].append(statistic(game, eq))
			count = count + 1
	game.reset()
	conf_intervals = [{} for x in xrange(0,len(equilibria))]
	count = 0

	#Find nth percentiles of each equilibrium's bootstrap regret distribution
	for boot_list in boot_lists:
		for interval in intervals:
			conf_intervals[count][interval] = np.percentile(boot_list,interval)
		count = count + 1
	return conf_intervals

def test():
	list = [0.3,0.5,0.9,1.3]
	print filter(lambda x: x>.7, list)[0]

def getAverages(payoff):
	#Get all possible averages of payoff data for a profile and the probability of each average
	distribution_cwr = list(CwR(payoff, r = len(payoff)))
	distribution_prod = list(product(payoff, repeat = len(payoff)))
	distribution_cwr = [list(np.sort(elem)) for elem in distribution_cwr]
	distribution_prod = [list(np.sort(elem)) for elem in distribution_prod]
	counts = []
	count = 0
	for elemc in distribution_cwr:
		for elemp in distribution_prod:
			if list(elemc) == elemp:
				count = count + 1
		counts.append(count)
		count = 0
	
	if np.count_nonzero(distribution_cwr) > 0:
		sum = np.sum(counts)
		counts = [elem/float(sum) for elem in counts]
	else:
		counts = [elem/elem for elem in counts]
	sum = len(distribution_cwr[0])
	distribution_cwr = [np.sum(elem)/sum for elem in distribution_cwr]
	return distribution_cwr,counts

def getCombinations(averages,averages_counts):
	#Get possible combinations of average where only averages with different amounts of payoff data are independent
	indices = []
	sizes = []
	curr_size = len(averages[0])
	curr_idx = 0
	for elem in averages:
		if(len(elem)!=curr_size):
			curr_idx = curr_idx + 1
			curr_size = len(elem)
		indices.append(curr_idx)
		sizes.append(len(elem))
	
	poss = []
	for size in np.unique(sizes):
		poss.append(list(xrange(size)))

	poss = list(product(*poss))
	
	combination = []
	count = 1
	combinations = []
	combo_counts = []
	for elem in poss:
		for avg,index in zip(averages,indices):
			combination.append(avg[elem[index]])
		for idx in np.unique(indices):
			count = count * averages_counts[idx][elem[idx]]
		combinations.append(combination)
		combo_counts.append(count)
		combination = []
		count = 1

	return combinations, combo_counts



def checkIntervals(game, equilibria, intervals=[95]):
	"""
	This function is just to test the results of the bootstrap function above
	"""
	
	print "--Manual Test--"
	
	#Find weights of payoffs for each equilibrium
	weights = []
	equilibria = [game.toArray(elem) for elem in equilibria]
	for equilibrium in equilibria:
		tmp = np.array(game.getWeights(equilibrium))
		weights.append(tmp)

	#Find all possible averages of the payoff data
	avgs = []
	avgs_counts= []
	prof_count = 0
	role_count = 0
	strat_count = 0
	for profile in game.counts:
		for role,number in zip(game.roles,profile):
			row = []
			for strategy,n in zip(game.strategies[role],number):
				payoff = game.getPayoffData(game.toProfile(profile),role,strategy)
				avg,avg_counts = getAverages(payoff)
				avgs.append(avg)
				avgs_counts.append(avg_counts)
				strat_count = strat_count + 1
			role_count = role_count + 1
		prof_count = prof_count + 1

	#Fix avgs_counts
	tmp = []
	current = -1
	for elem in avgs_counts:
		if not tmp and not elem.count(1)==len(elem):
			tmp.append(elem)
			current = current + 1
		if not elem.count(1)==len(elem) and not len(elem)==len(tmp[current]):
			tmp.append(elem)
			current = current + 1

	avgs_counts = tmp

	distributions = [[] for x in xrange(0,len(weights))]

	combinations,combo_counts = getCombinations(avgs,avgs_counts)


	#Find all possible combinations of the averages
	for combination,count in zip(combinations,combo_counts):
		combination = np.array(combination).reshape(prof_count,role_count/prof_count,strat_count/role_count)
		for x,weight,equilibrium in zip(xrange(0,len(weights)),weights,equilibria):
			strategy_vals = (combination*weight).sum(0)
			role_vals = (strategy_vals*equilibrium).sum(1)
			roles = []
			for role in game.roles:
				strategies = []
				for strategy in game.strategies[role]:
					r = game.index(role)
					s = game.index(role,strategy)
					strategies.append(strategy_vals[r][s] - role_vals[r])
				roles.append(max(strategies))
	#Calculate maximum regret for each equilibrium using each combination
			distributions[x].append(max(roles))

	#Find nth percentiles of each equilibrium's regret distribution
	conf_intervals = [{} for x in xrange(0,len(equilibria))]
	count = 0
	
	for distribution in distributions:
		probabilities = combo_counts
		sorted_lists = sorted(izip(distribution,probabilities),reverse = False, key = lambda x: x[0])
		distribution,probabilities = [[x[i] for x in sorted_lists] for i in range(2)]
		probabilities = np.cumsum(probabilities)
		for interval in intervals:
			idx = next(x[0] for x in enumerate(probabilities) if x[1] >= interval/100.0)
			percent  = distribution[idx]
			if(percent<0):
				percent = 0
			conf_intervals[count][interval] = percent
		count = count + 1
		
	print to_JSON_str(conf_intervals,indent=None)

def checkIntervalsAll(game, equilibria, intervals=[95]):
	"""
		This function is just to test the results of the bootstrap function above
		"""
	
	print "--Manual Test (All samples)--"
	
	#Find weights of payoffs for each equilibrium
	weights = []
	equilibria = [game.toArray(elem) for elem in equilibria]
	for equilibrium in equilibria:
		tmp = np.array(game.getWeights(equilibrium))
		weights.append(tmp)
	
	#Find all possible averages of the payoff data
	avgs = []
	prof_count = 0
	role_count = 0
	strat_count = 0
	for profile in game.counts:
		for role,number in zip(game.roles,profile):
			row = []
			for strategy,n in zip(game.strategies[role],number):
				payoff = game.getPayoffData(game.toProfile(profile),role,strategy)
				tmp = []
				if np.count_nonzero(payoff) > 0:
					combinations = list(product(payoff,repeat=len(payoff)))
					for combination in combinations:
						tmp.append(float(np.sum(combination))/float(len(combination)))
				else:
					tmp.append(0)
				avgs.append(tmp)
				strat_count = strat_count + 1
			role_count = role_count + 1
		prof_count = prof_count + 1
	
	distributions = [[] for x in xrange(0,len(weights))]

	#Find all possible combinations of the averages
	for combination in list(product(*(avgs))):
		combination = np.array(combination).reshape(prof_count,role_count/prof_count,strat_count/role_count)
		for x,weight,equilibrium in zip(xrange(0,len(weights)),weights,equilibria):
			strategy_vals = (combination*weight).sum(0)
			role_vals = (strategy_vals*equilibrium).sum(1)
			roles = []
			for role in game.roles:
				strategies = []
				for strategy in game.strategies[role]:
					r = game.index(role)
					s = game.index(role,strategy)
					strategies.append(strategy_vals[r][s] - role_vals[r])
				roles.append(max(strategies))
			#Calculate maximum regret for each equilibrium using each combination
			distributions[x].append(max(roles))


	#Find nth percentiles of each equilibrium's regret distribution
	conf_intervals = [{} for x in xrange(0,len(equilibria))]
	count = 0
	
	for distribution in distributions:
		for interval in intervals:
			percent  = np.percentile(distribution,interval)
			if(percent<0):
				percent = 0
			conf_intervals[count][interval] = percent
		count = count + 1

	print to_JSON_str(conf_intervals,indent=None)

def parse_args():
    parser = io_parser()
    parser.add_argument("profiles", type=str, help="File with profiles from"+\
        " input games for which confidence intervals should be calculated.")
    parser.add_argument("--interval", dest='intervals', metavar='INTERVALS',type = float, nargs = '+', help = "List of confidence intervals to calculate (default = [95])",default = [95])
    parser.add_argument("--point", metavar = 'POINTS', type = int, default = 1000, help = "Number of points to sample (default = 1000)")
    parser.add_argument("--check",action="store_true", help="Check the bootstrap confidence intervals manually (DO NOT USE for games larger than 3x2)")
    return parser.parse_args()


def main():
	args = parse_args()
	game = args.input
	intervals = args.intervals
	profiles = read(args.profiles)
	point = args.point
	if not isinstance(profiles, list):
		profiles = [profiles]
	if not isinstance(intervals, list):
		intervals = [intervals]
	#test()
	results = bootstrap(game,profiles,intervals,points = point)
	print to_JSON_str(results,indent=None)
	if args.check:
		checkIntervals(game,profiles,intervals)
		checkIntervalsAll(game,profiles,intervals)


if __name__ == "__main__":
	main()

