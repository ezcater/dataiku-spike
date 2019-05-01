# coding: utf-8

from base_predictor_version import BasePredictorVersion
from collections import defaultdict
from logger import logger
import math as math
import numpy as np
import os
import pandas as pd
import patsy as patsy
import xgboost as xgb
import sklearn.model_selection as sklearn
from iter_flatten import iter_flatten


class UserQualityVersion(BasePredictorVersion):
  
  def __init__(self, base_predictor, param_list):
    self.base_predictor = base_predictor
    self.config = param_list['model']
    self.channel = param_list['channel']
    self.files = defaultdict(dict)
    self.metrics_tags = [
      "day_horizon:{}".format(self.config["day_horizon"]),
      "version:{}".format(self.config["version"]),
      "channel:{}".format(self.channel)
    ]
    self._s3_conn = None
  
  def run(self):
    self.filter_data_and_create_design_matrices()
    self.train_test_split()
    self.train_and_predict()
  
  @staticmethod
  def generate_iterations(config):
    param_list = []
    for model in config['predictions']:
      for channel in model['channel']:
        param_list.append({'model': model, 'channel': channel})
    return param_list
  
  @staticmethod
  def accuracy_metrics(acc, name):
    pos = acc[acc.label > 0]
    return {
      'subset': name,
      'bias': np.mean(acc['predicted_value']) - np.mean(acc['label']),
      'relative_error': np.mean((pos['predicted_value'] - pos['label']) / pos['label']),
      'rmse': math.sqrt(np.mean(np.square(acc['predicted_value'] - acc['label']))),
      'user_count': acc.shape[0]
    }
  
  @classmethod
  def generate_accuracy_suite(cls, data):
    metric_list = []
    metric_list.append(cls.accuracy_metrics(data, 'full_data'))
    metric_list.append(cls.accuracy_metrics(data.loc[data['label'] < 5000], 'sub_5000'))
    metric_list.append(cls.accuracy_metrics(data.loc[data['label'] >= 5000], 'super_5000'))
    return pd.DataFrame(metric_list)
  
  @staticmethod
  def add_static_accuracy_columns(data, config, date_stamp, channel):
    data['algorithm'] = config['name']
    data['target_purpose'] = config['target_purpose']
    data['target_value'] = config['goal_column']
    data['day_horizon'] = config['day_horizon']
    data['version'] = config['version']
    data['model_date'] = date_stamp
    data['channel'] = channel
  
  @staticmethod
  def add_static_output_columns(data, config):
    data['algorithm'] = config['name']
    data['predicted_breakeven_likelihood'] = 0
    data['day_horizon'] = config['day_horizon']
    data['official_flag'] = config['official_flag']
    data['target_purpose'] = config['target_purpose']
    data['target_value'] = config['goal_column']
    data['version'] = config['version']
  
  @staticmethod
  def clip_predictions(preds, bookings_at_predict_time):
    # highest of prediction, bookings on the model version's day horizon, and 0. guarantees non-negative results
    return np.maximum.reduce([preds, bookings_at_predict_time, np.full(preds.size, 0)])
  
  def filter_data_and_create_design_matrices(self):
    data_for_training = self.base_predictor.input_data.copy(deep=True)
    data_for_prediction = self.base_predictor.input_data.copy(deep=True)
    
    if self.channel == 'all':
      data_for_training = data_for_training.loc[
        data_for_training['days_since_first_order'] >= self.config['goal_horizon']
      ]
      data_for_prediction = data_for_prediction.loc[
        data_for_prediction['days_since_first_order'] >= self.config['day_horizon']
      ]
    else:
      data_for_training = data_for_training.loc[
        (data_for_training['days_since_first_order'] >= self.config['goal_horizon']) &
        (data_for_training['attribution_level_1'] == self.channel)
      ]
      data_for_prediction = data_for_prediction.loc[
        (data_for_prediction['days_since_first_order'] >= self.config['day_horizon']) &
        (data_for_prediction['attribution_level_1'] == self.channel)
      ]
    
    shuffled_training_data = data_for_training.sample(frac=1)
    
    training_columns = [i for i in iter_flatten(self.config['training_columns'])]
    
    self.full_training_labels, filtered_training_data = patsy.dmatrices(
      self.config['goal_column'] + ' ~ 0 + ' + ' + '.join(training_columns),
      data=shuffled_training_data,
      return_type="dataframe"
    )
    # fix enum column headers for xgb input requirements
    self.filtered_training_data = filtered_training_data.rename(columns=lambda x: x.replace("[", "(").replace("]", ")"))
    
    filtered_prediction_data = patsy.dmatrix(
      '0 + ' + ' + '.join(training_columns),
      data=data_for_prediction,
      return_type="dataframe"
    )
    # fix enum column headers for xgb input requirements
    self.filtered_prediction_data = filtered_prediction_data.rename(
      columns=lambda x: x.replace("[", "(").replace("]", ")"))
  
  def train_test_split(self):
    if self.config['stratify_flag']:
      # bins = np.linspace(0, self.full_training_labels.max()+1, num=5)
      manual_bins = np.array(self.config['stratify_bins'])
      label_bins = np.digitize(self.full_training_labels, manual_bins)
    else:
      label_bins = None  # default behavior is to not stratify
    
    self.training_data, self.testing_data, self.training_labels, self.testing_labels = sklearn.train_test_split(
      self.filtered_training_data, self.full_training_labels, train_size=self.config['percent_training'], stratify=label_bins)
  
  def train_and_predict(self, default_xg_iterations=20):
    # train model, save predictions
    xg_training_data = xgb.DMatrix(self.training_data, label=self.training_labels)
    self.model = xgb.train(self.config['xgb_params'], xg_training_data, default_xg_iterations)
    testing_preds = self.model.predict(xgb.DMatrix(self.testing_data))
    algo_output_preds = self.model.predict(xgb.DMatrix(self.filtered_prediction_data))
    
    # re-associate labels with full data
    self.testing_data.loc[:, 'label'] = self.testing_labels.values
    self.filtered_prediction_data = self.filtered_prediction_data.join(self.full_training_labels, how="left")
    
    # associate predictions with full data
    self.testing_data['predicted_value'] = self.clip_predictions(testing_preds,
                                                                 self.testing_data.loc[:, self.config['min_value_column']])
    self.filtered_prediction_data['predicted_value'] = self.clip_predictions(
      algo_output_preds, self.filtered_prediction_data[self.config['min_value_column']])
    
    # adjust columns for output of predictions
    self.output_data = self.filtered_prediction_data[['predicted_value', self.config['goal_column']]].copy(deep=True)
    self.output_data.rename(columns={self.config['goal_column']: 'actual_value'},
                            inplace=True)
    self.add_static_output_columns(self.output_data, self.config)
    
    # calculate accuracy metrics
    self.accuracy = self.generate_accuracy_suite(self.testing_data)
    
    # adjust columns for output of accuracy
    self.add_static_accuracy_columns(self.accuracy, self.config, self.base_predictor.date_stamp, self.channel)
  
  def generate_metrics(self):
    metrics = {}
    
    metrics["total_predictions"] = self.output_data.shape[0]
    metrics["average_predicted_vale"] = self.output_data["predicted_value"].mean()
    metrics["below_zero_predictions"] = sum(self.output_data["predicted_value"] < 0)
    metrics["relative_error"] = self.accuracy.at[0, "relative_error"]
    metrics["rmse"] = self.accuracy.at[0, "rmse"]
    metrics["bias"] = self.accuracy.at[0, "bias"]
    
    return metrics
  
  def validate_data(self):
    errors = []
    
    if self.output_data.shape[1] != 9:
      errors.append("Output data should have 9 columns; has {}".format(self.output_data.shape[1]))
    if self.accuracy.shape[1] != 12:
      errors.append("Accuracy data should have 12 columns; has {}".format(self.accuracy.shape[1]))
    
    if self.training_data.shape[0] == 0:
      errors.append("Training data has no rows")
    if self.output_data.shape[0] == 0:
      errors.append("Output data has no rows")
    if self.accuracy.shape[0] == 0:
      errors.append("Accuracy data has no rows")
    
    if self.output_data['predicted_value'].max() > 1000000:
      errors.append("Output data has a prediction larger than $1,000,000")
    
    if np.isin(True, self.output_data.index.duplicated()).item():
      errors.append("Output data has a duplicated user ID")
    
    return errors
  
  def setup_file_data(self):
    for output_type in ['model', 'training_data', 'output_data', 'accuracy']:
      # populating a defaultdict to map file names to file paths
      self.files[output_type]['name'] = '{}_{}'.format(output_type, self.config['name'] + '_' + self.channel)
      self.files[output_type]['path'] = os.path.join(self.base_predictor.output_path,
                                                     self.files[output_type]['name'])
  
  def save_local_files(self):
    if len(self.files) == 0:
      self.setup_file_data()
    
    logger.info("Writing model to file", extra={"path": self.files['model']['path']})
    self.model.save_model(self.files['model']['path'])
    
    logger.info("Writing training_data to file", extra={"path": self.files['training_data']['path']})
    self.training_data.to_csv(self.files['training_data']['path'])
    
    logger.info("Writing output_data to file", extra={"path": self.files['output_data']['path']})
    self.output_data.to_csv(self.files['output_data']['path'])
    
    logger.info("Writing accuracy to file", extra={"path": self.files['accuracy']['path']})
    self.accuracy.to_csv(self.files['accuracy']['path'], index=False)
  
  def upload_to_s3(self):
    if len(self.files) == 0:
      self.setup_file_data()
    
    # uploading model to S3
    model_s3_filename = os.path.join(self.base_predictor.s3_key, self.files['model']['name'])
    logger.info("Uploading model to S3", extra={"key": model_s3_filename})
    self.warehouse_bucket().upload_file(self.files['model']['path'], model_s3_filename)
    
    # uploading training data to S3
    training_data_s3_filename = os.path.join(self.base_predictor.s3_key, self.files['training_data']['name'])
    logger.info("Uploading training_data to S3", extra={"key": training_data_s3_filename})
    self.warehouse_bucket().upload_file(self.files['training_data']['path'],
                                        training_data_s3_filename)
    
    # uploading output data to S3
    output_data_s3_filename = os.path.join('machine-learning', 'output_estimates_python',
                                           self.files['output_data']['name'])
    logger.info("Uploading output_data to S3", extra={"key": output_data_s3_filename})
    self.warehouse_bucket().upload_file(self.files['output_data']['path'],
                                        output_data_s3_filename)
    
    # uploading accuracy to S3
    accuracy_s3_filename = os.path.join('machine-learning', 'output_accuracies_python',
                                        self.base_predictor.date_stamp,
                                        self.files['accuracy']['name'])
    logger.info("Uploading accuracy to S3", extra={"key": accuracy_s3_filename})
    self.warehouse_bucket().upload_file(self.files['accuracy']['path'], accuracy_s3_filename)
