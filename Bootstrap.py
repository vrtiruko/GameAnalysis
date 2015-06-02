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

from itertools import product, combinations_with_replacement as CwR, repeat
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
	boot_lists = [[] for x in xrange(0,len(equilibria))]
	method = getattr(game, method)
	for __ in range(points):
		method(*method_args)
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

def checkIntervals(game, equilibria, intervals=[95]):
	"""
	This function is just to test the results of the bootstrap function above
	"""
	
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
				combinations = list(product(payoff,repeat=len(payoff)))
				tmp = []
				for combination in combinations:
					tmp.append(float(np.sum(combination))/float(len(combination)))
				avgs.append(tmp)
				strat_count = strat_count + 1
			role_count = role_count + 1
		prof_count = prof_count + 1

	distributions = [[] for x in xrange(0,len(weights))]

	#Find all possible combinations of the averages
	for combination in list(product(*(avgs))):
		combination = np.array(combination).reshape(prof_count,role_count/prof_count,strat_count/prof_count)
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
	for distribution in distributions:
		for interval in intervals:
			percent = np.percentile(distribution,interval)
			if(percent<0):
				percent = 0
			print str(interval) + " " + str(percent)
		
	"""
	combinations[0] = np.array(combinations[0])
	combinations[0] = combinations[0].reshape(prof_count,role_count/prof_count,strat_count/prof_count)
	strategy_vals = (combinations[0]*weights[1]).sum(0)
	print strategy_vals
	role_vals = (strategy_vals*equilibria[1]).sum(1)
	print role_vals
	roles = []
	for role in game.roles:
		strategies = []
		for strategy in game.strategies[role]:
			r = game.index(role)
			s = game.index(role,strategy)
			strategies.append(strategy_vals[r][s] - role_vals[r])
		roles.append(max(strategies))
	print max(roles)


	# Regret for 100% Bravo
	# 2x2 Game
	#combinations = list(product(avgs[2][0],avgs[1][0]))
	# 3x2 Game
	combinations = list(product(avgs[3][0],avgs[2][0]))
	distribution = []
	for combination in combinations:
		distribution.append(combination[1]-combination[0])
	for interval in intervals:
		percent = np.percentile(distribution,interval)
		if(percent<0):
			percent = 0
		print str(interval) + " " + str(percent)

	# Regret for 100% Alpha
	combinations = list(product(avgs[0][0],avgs[1][1]))
	distribution = []
	for combination in combinations:
		distribution.append(combination[1]-combination[0])
	for interval in intervals:
		percent = np.percentile(distribution,interval)
		if(percent<0):
			percent = 0
		print str(interval) + " " + str(percent)

	#Regret for 50% Alpha 50% Bravo
		#Generate lists of averages
	#2x2 Game
	#combinations_all = list(product(avgs[1][1],avgs[0][0],avgs[1][0],avgs[2][0]))
	#3x2 Game
	combinations_all = list(product(avgs[0][0],avgs[0][1],avgs[1][0],avgs[1][1],avgs[2][0],avgs[2][1],avgs[3][0],avgs[3][1]))
	print combinations_all[0]
	print combinations_all[0]*weights[2]
	distribution = []
	for combination in combinations_all:
		distribution.append(np.max([float(np.mean([combination[1],combination[2]])),float(np.mean([combination[0],combination[3]]))])-float(np.mean(combination)))

	for interval in intervals:
		percent = np.percentile(distribution,interval)
		if(percent<0):
			percent = 0
		print str(interval) + " " + str(percent)
	"""

def parse_args():
    parser = io_parser()
    parser.add_argument("profiles", type=str, help="File with profiles from"+\
        " input games for which confidence intervals should be calculated.")
    parser.add_argument("--interval", dest='intervals', metavar='INTERVALS',type = float, nargs = '+', help = "List of confidence intervals to calculate (default = [95])",default = [95])
    parser.add_argument("--point", metavar = 'POINTS', type = int, default = 1000, help = "Number of points to sample (default = 1000)")
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
	results = bootstrap(game,profiles,intervals,points = point)
	print to_JSON_str(results,indent=None)
	checkIntervals(game,profiles, intervals)


if __name__ == "__main__":
	main()

