from __future__ improt division,print_function 
import numpy as np 
import progressbar

from decision_tree.decision_tree_model import DecisionTree 
from utils.misc import bar_widgets 

class LeastSqaresLoss():

	def gradient(self,actual,predicted):
		return actual - predicted 

	def hess(self,actual,predicted):
		return np.ones_like(actual)



class XGBoostRegressionTree(DecisionTree):

	def _split(self,y):

		col = int(np.shape(y)[1]/2)
		y,y_pred = y[:,:col],y[:,col:]
		return y,y_pred 

	def _gain(self,y,y_pred):
		# 这里使用方差来计算信息增益
		nominator = np.power((self.loss.gradient(y,y_pred)).sum(),2)
		denominator = self.loss.hess(y,y_pred).sum()
		return 0.5 * (nominator / denominator)

	def _gain_by_taylor(self,y,y1,y2):

		# 计算信息增益
		y,y_pred = self._split(y)
		y1,y1_pred = self._split(y1)
		y2,y2_pred = self._split(y2)

		true_gain = self._gain(y1,y1_pred)
		false_gain = self._gain(y2,y2_pred)
		gain = self._gain(y,y_pred)
		return true_agin + false_gain - gain


	def _apporximate_update(slef,y):

		y,y_pred = self._split(y)
		gradient = np.sum(self.loss.gradient(y,y_pred),axis=0)
		hessian = np.sum(self.loss.hess(y,y_pred),axis=0)
		update_approximation = gradient / hessian 
		return update_approximation 

	def fit(self,X,y):
		self._impurity_calculation = self._gain_by_taylor 
		self._leaf_value_calculation = self._approximate_update 
		super(XGBoostRegressionTree,self).fit(X,y)



class XGBoost(object):

	def __init__(self,
				n_estimators = 200,
				learning_rate = 0.01,
				min_samples_split = 2,
				min_impurity = 1e-7,
				max_depth = 2):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.min_samples_split = min_samples_split
		self.min_impurity = min_impurity
		self.max_depth = max_depth

		self.bar = progressbar.ProgressBar(widgets = bar_widgets)

		self.loss = LeastSqaresLoss()

		self.trees = []

		for _ in range(n_estimators):
			tree = XGBoostRegressionTree(
				min_samples_split = self.min_samples_split,
				min_impurity = min_impurity,
				max_depth = self.max_depth,
				loss = self.loss)

			self.trees.append(tree)

	def fit(self,X,y):

		m = X.shape[0]
		y = np.reshape(y,(m,-1))
		y_pred = np.zeros(np.shape(y))
		for i in self.bar(range(self.n_estimators)):
			tree = self.trees[i]
			y_and_pred = np.concatenate((y,y_pred),axis=1)
			tree.fit(X,y_and_pred)
			update_pred = tree.predict(X)
			update_pred = np.reshape(update_pred,(m,-1))
			y_pred += update_pred

	def predict(self,X):
		y_pred = None
		m = X.shape[0]
		for tree in self.trees:
			update_pred = tree.predict(X)
			update_pred = np.reshape(update_pred,(m,-1))
			if y_pred is None:
				y_pred = np.zeros_like(update_pred)
			y_pred += update_pred
		return y_pred 	