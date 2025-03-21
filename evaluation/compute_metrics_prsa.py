import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from loguru import logger
import code

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class PrsaEvaluator():

	def __init__(self, scaler_pm10, scaler_pm25):

		self.scaler_pm10 = scaler_pm10
		self.scaler_pm25 = scaler_pm25

	def compute_metrics_prsa(self, eval_preds):

		# logger.info(eval_preds)
		labels = eval_preds.label_ids
		preds = eval_preds.predictions
		# logger.info(labels)
		# logger.info(preds)
		# code.interact(local=locals())

		assert labels.shape == preds.shape
		logger.info(f'labels and preds shape during evaluation is {labels.shape}')

		preds_pm25 = preds[:,:,0]
		preds_pm10 = preds[:,:,1]
		preds_pm25 = self.scaler_pm25.inverse_transform(preds_pm25)
		preds_pm10 = self.scaler_pm10.inverse_transform(preds_pm10)
		preds[:,:,0] = preds_pm25
		preds[:,:,1] = preds_pm10
		preds = preds.flatten()
		preds_pm10 = preds_pm10.flatten()
		preds_pm25 = preds_pm25.flatten()
		labels_pm25 = labels[:,:,0]
		labels_pm10 = labels[:,:,1]
		labels_pm25 = self.scaler_pm25.inverse_transform(labels_pm25)
		labels_pm10 = self.scaler_pm10.inverse_transform(labels_pm10)
		labels[:,:,0] = labels_pm25
		labels[:,:,1] = labels_pm10
		labels = labels.flatten()
		labels_pm10 = labels_pm10.flatten()
		labels_pm25 = labels_pm25.flatten()

		mae = mean_absolute_error(labels, preds)
		mae_perc = mean_absolute_percentage_error(labels, preds)
		rmse = mean_squared_error(labels, preds, squared=False)

		mae_pm25 = mean_absolute_error(labels_pm25, preds_pm25)
		mae_perc_pm25 = mean_absolute_percentage_error(labels_pm25, preds_pm25)
		rmse_pm25 = mean_squared_error(labels_pm25, preds_pm25, squared=False)
		mae_pm10 = mean_absolute_error(labels_pm10, preds_pm10)
		mae_perc_pm10 = mean_absolute_percentage_error(labels_pm10, preds_pm10)
		rmse_pm10 = mean_squared_error(labels_pm10, preds_pm10, squared=False)

		count_labels_tot = labels.shape[0]
		count_labels_pm25 = labels_pm25.shape[0]
		count_labels_pm10 = labels_pm10.shape[0]
		count_preds_tot = preds.shape[0]
		count_preds_pm25 = preds_pm25.shape[0]
		count_preds_pm10 = preds_pm10.shape[0]
		mean_labels_tot = np.mean(labels)
		mean_labels_pm25 = np.mean(labels_pm25)
		mean_labels_pm10 = np.mean(labels_pm10)
		mean_preds_tot = np.mean(preds)
		mean_preds_pm25 = np.mean(preds_pm25)
		mean_preds_pm10 = np.mean(preds_pm10)

		return {
			'Mae_tot': mae,
			'Mape_tot': mae_perc,
			'Rmse_tot': rmse,
			'Mae_25': mae_pm25,
			'Mape_25': mae_perc_pm25,
			'Rmse_25': rmse_pm25,
			'Mae_10': mae_pm10,
			'Mape_10': mae_perc_pm10,
			'Rmse_10': rmse_pm10,
			'count_labels_tot': count_labels_tot,
			'count_labels_pm25': count_labels_pm25,
			'count_labels_pm10': count_labels_pm10,
			'count_preds_tot': count_preds_tot,
			'count_preds_pm25': count_preds_pm25,
			'count_preds_pm10': count_preds_pm10,
			'mean_labels_tot': mean_labels_tot,
			'mean_labels_pm25': mean_labels_pm25,
			'mean_labels_pm10': mean_labels_pm10,
			'mean_preds_tot': mean_preds_tot,
			'mean_preds_pm25': mean_preds_pm25,
			'mean_preds_pm10': mean_preds_pm10,
		}
