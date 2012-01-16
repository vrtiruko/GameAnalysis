import numpy as np

from math import factorial
from operator import mul, eq
from itertools import product, combinations_with_replacement as CwR

def prod(collection):
	"""
	Product of all elements in the collection.
	elements must support multiplication
	"""
	return reduce(mul, collection)


def nCr(n,k):
	"""
	Number of combinations: n choose k.
	"""
	return prod(range(n-k+1,n+1)) / factorial(k)


def game_size(n,s):
	"""
	Number of profiles in a symmetric game with n players and s strategies.
	"""
	return nCr(n+s-1,n)


def profile_repetitions(p):
	"""
	Number of normal form profiles that correspond to a role-symmetric profile.
	"""
	return prod([factorial(sum(row)) / prod(map(factorial, row)) for row in p])


def list_repr(l, sep=", "):
	"""
	Creates a string representation of the elements of a collection.
	"""
	try:
		return reduce(lambda x,y: str(x) + sep + str(y), l)
	except TypeError:
		return ""


class Array(np.ndarray):
	"""
	An immutable (and therefore hashable and sortable) subclass of ndarray.

	Useful for profile, payoff, and mixed strategy data.
	"""

	def _blocked_attribute(obj):
		raise TypeError("hashing requires immutability")
	_blocked_attribute = property(_blocked_attribute)
	byteswap = fill = itemset = resize = setasflat = setfield = setflags = \
			sort = _blocked_attribute

	def __new__(cls, *args, **kwargs):
		a = np.array(*args, **kwargs)
		a = np.ndarray.__new__(cls, shape=a.shape, dtype=a.dtype, \
				strides=a.strides, buffer=a.data)
		cls.setflags = np.ndarray.setflags
		a.setflags(write=False)
		cls.setflags = cls._blocked_attribute
		return a

	def __hash__(self):
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(tuple(self))
			return self._hash

	def __eq__(self, other):
		return np.array_equal(self, other)

	def __cmp__(self, other):
		assert self.shape == other.shape
		try:
			i = self != other
			return self[i][0] - other[i][0]
		except IndexError:
			return False

	def __lt__(self, other):
		return min(self.__cmp__(other), 0)

	def __gt__(self, other):
		return max(self.__cmp__(other), 0)

	def __le__(self, other):
		if self == other:
			return True
		return self < other

	def __ge__(self, other):
		if self == other:
			return True
		return self > other


class Profile(Array):
	def __new__(cls, *args, **kwargs):
		kwargs['dtype'] = int
		game = kwargs.pop('game')
		p = Array.__new__(cls, *args, **kwargs)
		p.game = game
		p.reps = profile_repetitions(p)
		p.dev_reps = game.zeros(dtype=int)
		for i, r in enumerate(game.roles):
			for j, s in enumerate(game.strategies[r]):
				if p[i,j] > 0:
					opp_prof = p - game.array_index(r,s)
					p.dev_reps[i,j] = profile_repetitions(opp_prof)
				else:
					p.dev_reps[i,j] = 0
		return p

	def neighbors(self, role=None, strategy=None):
		if role == None:
			return sum([self.neighbors(r) for r in self.game.roles], [])
		if strategy == None:
			return sum([self.neighbors(role, s) for s in \
					self.game.strategies[role]], [])
		role_index = self.game.index(role)
		strategy_index = self.game.index(role, strategy)
		neighbors = []
		for i in range(len(self.game.strategies[role])):
			if i == strategy_index:
				continue
			p = np.array(self)
			p[role_index, strategy_index] -= 1
			p[role_index, i] += 1
			neighbors.add(p)
		return neighbors

	def dist(self, other):
		"""
		L1-norm/2 gives the number of deviations by which the profiles differ
		"""
		return np.linalg.norm(self - other, 1) / 2

	def __str__(self):
		return repr(self)

	def __repr__(self):
		return "{" + list_repr([r +": ("+ list_repr([str(c) +"x "+s for c,s \
				in filter(lambda x: x[0], zip(self[i], self.game.strategies[ \
				r]))]) +")" for i,r in enumerate(self.game.roles)]) + "}"


class Mixture(Array):
	def __new__(cls, probabilities, game):
		a = np.array(probabilities, dtype=float).clip(0)
		a[a.max(1) == 0] = 1
		a[game.mask] = 0
		a = Array.__new__(cls, (a.T/a.sum(1)).T)
		a.game = game
		return a

	def probability(self, profile):
		return prod((self ** profile).flat) * profile.reps

	def dist(self, other):
		"""
		L2-norm gives the euclidian distance between mixture vectors
		"""
		return np.linalg.norm(self - other, 2)

	def __str__(self):
		return repr(self)

	def __repr__(self):
		string = ""
		for i,r in enumerate(self.game.roles):
			string += r + ": {"
			for j,s in enumerate(self.game.strategies[r]):
				if self[i,j] > 0:
					string += s + ":" + str(round(100*self[i,j], 1)) + "%, "
			string = string[:-2] + "}; "
		return string[:-2]


class Game(dict):
	def __init__(self, roles=[], counts={}, strategies={}, payoff_data=[]):
		"""
		Role-symmetric game representation.

		__init__ parameters:
		roles: collection of role-name strings
		counts: mapping from roles to number of players per role
		strategies: mapping from roles to per-role strategy sets
		payoff_data: collection of data objects mapping roles to collections
				of (strategy, count, value) tuples
		"""
		self.roles = sorted(set(map(str, roles)))
		self.counts = {r : int(counts[r]) for r in self.roles}
		self.strategies = {r : sorted(set(map(str, strategies[r]))) \
				for r in self.roles}

		self.numStrategies = [len(self.strategies[r]) for r in self.roles]
		self.maxStrategies = max(self.numStrategies)
		self.mask = np.array([[False]*s + [True]*(self.maxStrategies - s) for \
				s in self.numStrategies])
		self.size = prod([game_size(self.counts[r], self.numStrategies[i]) \
				for i,r in enumerate(self.roles)])
		self.role_index = {r:i for i,r in enumerate(self.roles)}
		self.strategy_index = {r : {s:i for i,s in enumerate( \
				self.strategies[r]) } for r in self.roles}

		for datum in payoff_data:
			prof = self.zeros()
			payoffs = self.zeros()
			for role_index, role in enumerate(self.roles):
				for strategy, count, value in datum[role]:
					strategy_index = self.index(role, strategy)
					prof[role_index, strategy_index] = count
					payoffs[role_index, strategy_index] = value
			self[Profile(prof, game=self)] = Array(payoffs, dtype=float)

	def zeros(self, dtype=float, masked=True):
		zeros = np.zeros([len(self.roles), self.maxStrategies], dtype=dtype)
		if masked:
			return np.ma.array(zeros, mask=self.mask)
		return zeros

	def index(self, role, strategy=None):
		"""
		index(r) returns the role-index of r
		index(r,s) returns the strategy index of s (for role r)
		"""
		if strategy != None:
			return self.strategy_index[role][strategy]
		return self.role_index[role]

	def array_index(self, role, strategy=None):
		"""
		array_index(r,s) returns a boolean ndarray version of index(r,s)
		"""
		a = self.zeros(dtype=bool, masked=False)
		if strategy == None:
			a[self.index(role)] += 1
		else:
			a[self.index(role), self.index(role, strategy)] += 1
		return a

	def getPayoff(self, profile, role, strategy):
		return self[profile][self.index(role)][self.index(role, strategy)]

	def expectedValues(self, mix):
		return sum([self[p] * (mix**p).prod() * p.dev_reps / (mix + \
				np.finfo(np.float64).tiny) for p in self])

	def allProfiles(self):
		return [Profile([[list(p[i]).count(s) for s in self.strategies[r]] + \
				[0]*(self.maxStrategies - self.numStrategies[i]) for i,r in \
				enumerate(self.roles)], game=self) for p in product(*[CwR( \
				self.strategies[r], self.counts[r]) for r in self.roles])]

	def knownProfiles(self):
		return self.keys()

	def uniformMixture(self):
		return Mixture(self.zeros(), self)

	def biasedMixture(self):
		raise NotImplementedError

	def __cmp__(self, other):
		return cmp(self.roles, other.roles) or \
				cmp(self.counts, other.counts) or \
				cmp(self.strategies, other.strategies) or \
				dict.__cmp__(self, other)

	def __repr__(self):
		return ("RoleSymmetricGame:\n\troles: " + list_repr(self.roles) + \
				"\n\tcounts:\n\t\t" + list_repr(map(lambda x: str(x[1]) +"x "+ \
				str(x[0]), sorted(self.counts.items())), sep="\n\t\t") + \
				"\n\tstrategies:\n\t\t" + list_repr(map(lambda x: x[0] + \
				":\n\t\t\t" + list_repr(x[1], sep="\n\t\t\t"), \
				sorted(self.strategies.items())), sep="\n\t\t") + \
				"\npayoff data for " + str(len(self)) + " out of " + \
				str(self.size) + " profiles").expandtabs(4)

	def subgame(self, roles=[], strategies={}):
		raise NotImplementedError

	def isSubgame(self, big_game):
		raise NotImplementedError

	def regret(self, profile, strategy=None):
		raise NotImplementedError

	def mixtureRegret(self, mix, strategy=None):
		raise NotImplementedError
