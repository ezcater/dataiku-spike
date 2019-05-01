from collections import defaultdict
from logger import logger
import os
import boto3


class BasePredictorVersion():
  def __init__(self, base_predictor, param_list):
    self.base_predictor = base_predictor
    self.config = param_list
    self.files = defaultdict(dict)
    self._s3_conn = None
  
  @staticmethod
  def generate_iterations(config):
    return config['predictions']
  
  def generate_metrics(self):
    logger.info("No metrics set up for {}".format(type(self).__name__))
    return {}
  
  def validate_data(self):
    logger.info("No data validation set up for {}".format(type(self).__name__))
    return []
  
  def s3_conn(self):
    self._s3_conn = self._s3_conn or boto3.resource('s3')
    return self._s3_conn
  
  def warehouse_bucket(self):
    return self.s3_conn().Bucket(os.environ['EZCATER_WAREHOUSE_BUCKET'])
