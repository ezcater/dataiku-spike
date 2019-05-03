# coding: utf-8

from datadog import statsd
from datetime import datetime
from logger import logger
import datadog
import importlib
import os
import pandas as pd
import sentry
import slack
import sys
import warehouse
import yaml


class BasePredictor():
  def __init__(self,
          predictor_type,
          sql_config_filename=None,
          predictor_config_filename=None,
          statsd_time_name=None,
          s3_key=None):
    """
    The BasePredictor is meant to set up the foundation for any type of machine learning predictions

    :param predictor_type (str): string name predictor. used in auto-naming other files
    :param sql_config_filename (str): yaml file name
    :param predictor_config_filename (str): yaml file name
    :param statsd_time_name (str): name used by statsd to time the run
    :param s3_key (str): name of key used to send data to s3
    """
    
    self.predictor_type = predictor_type
    self.sql_config_filename = sql_config_filename or "sql_config_{}.yml".format(self.predictor_type)
    self.predictor_config_filename = predictor_config_filename or (f"predictor_config_{self.predictor_type}" + ".yml")
    
    self.statsd_time_name = statsd_time_name or "sky-py.{}.time".format(self.predictor_type)
    self.s3_key = s3_key or "{}".format(self.predictor_type)
    
    self.configure_environment()
    
    datadog.initialize(statsd_host='datadog.kube-system', statsd_port=8125)
    
    self.pull_data()
  
  def version_class(self):
    module = importlib.import_module(self.pred_config["class_module"])
    return getattr(module, self.pred_config["class_name"])
  
  def configure_environment(self):
    self.conn = warehouse.connect()
    self.date_stamp = datetime.now().strftime("%Y-%m-%d")
    
    # local output
    self.output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'output',
                                    self.predictor_type,
                                    datetime.today().isoformat())
    logger.info("Creating output directory", extra={"path": self.output_path})
    os.makedirs(self.output_path)
  
  def pull_data(self):
    # get config data
    with open(self.predictor_config_filename, 'r') as ymlfile:
      self.pred_config = yaml.load(ymlfile)
    
    # get sql data
    with open(self.sql_config_filename, 'r') as ymlfile:
      input_data_pull = yaml.load(ymlfile)['sql']
    self.input_data = pd.read_sql_query(input_data_pull, self.conn)
    if self.pred_config.get('index_value') is not None:
      self.input_data = self.input_data.set_index(self.pred_config['index_value'])
      
    # get secondary sql -- TODO: generalize to any number of secondary sql pulls
    if 'extra_sql' in self.pred_config.keys():
      with open(self.pred_config['extra_sql'], 'r') as ymlfile:
        input_data_pull = yaml.load(ymlfile)['sql']
      self.extra_sql = pd.read_sql_query(input_data_pull, self.conn)
    
  
  def run(self, send_to_s3_override=False):
    
    # staging S3 files still get consumed by the ETL
    # in most cases, we wouldn't want this in local development
    send_to_s3 = send_to_s3_override or os.environ.get('SEND_TO_S3') == 'True'
    
    # sample a subset of yml if testing
    if os.environ.get("CALLED_FROM_TEST", None) == "True":
      sampling_attributes = self.pred_config.get("test_sampling_attributes", None)
      self.pred_config['predictions'] = self.sample_pred_config(
        self.pred_config['predictions'], sampling_attributes=sampling_attributes
      )
    
    # loop and run
    for params in self.version_class().generate_iterations(self.pred_config):
      self.run_predictor_version(params, send_to_s3)
  
  def run_predictor_version(self, params, send_to_s3):
    version = self.version_class()(self, params)
    version.run()
    
    self.metrics_to_datadog(version)
    
    self.output_data = version.output_data
    
    # version.save_local_files()
    
    # # export local files out to s3
    # if not self.validate_data(version) and send_to_s3 and os.environ.get('EZCATER_WAREHOUSE_BUCKET'):
    #   version.upload_to_s3()
  
  def run_and_notify(self, send_to_s3_override=False):
    try:
      slack.send("{} started".format(self.s3_key), channel_name="sky-py")
      with statsd.timed(self.statsd_time_name):
        self.run(send_to_s3_override=send_to_s3_override)
      statsd.increment(self.statsd_time_name, tags=['success'])
      slack.send("{} finished".format(self.s3_key), channel_name="sky-py")
    except Exception:
      statsd.increment(self.statsd_time_name, tags=['error'])
      slack.send("{} failed".format(self.s3_key), channel_name="sky-py")
      sentry.handleException(sys.exc_info())
  
  def validate_data(self, version):
    errors = version.validate_data()
    
    if errors:
      slack.send(
        "Skipping S3 upload for {} due to the following errors:\n".format(version.config["name"]) + "\n".join(errors),
        channel_name="sky-py")
    
    return errors
  
  def metrics_to_datadog(self, version):
    return
    # def update_gauge(title, value):
    #   statsd.gauge("sky_py.{}.{}".format(self.predictor_type, title), value,
    #                tags=version.metrics_tags)
    #
    # metrics = version.generate_metrics()
    # for key, value in metrics.items():
    #   update_gauge(key, value)
  
  def sample_pred_config(self, initial_config, sampling_attributes=None):
    def sample_one_of_each(df, col_name):
      return df.groupby(col_name).apply(lambda df: df.sample(1))
    
    if not sampling_attributes:
      logger.info("No test sampling done for {}".format(self.version_class()))
      return initial_config
    
    config_df = pd.DataFrame(initial_config)
    samples = []
    
    # get a small set of predictions to run tests on, at least one for each value of each attribute passed in
    for grouping_var in sampling_attributes:
      if isinstance(grouping_var, str):
        samples.append(sample_one_of_each(config_df, grouping_var))
      elif isinstance(grouping_var, list):
        cloned_df = config_df.copy(deep=True)
        cloned_df['grouping_var'] = cloned_df[grouping_var].apply(lambda df: ''.join(df.astype(str)), axis=1)
        sampled_df = sample_one_of_each(cloned_df, 'grouping_var')
        samples.append(sampled_df.drop(['grouping_var'], axis=1))
    
    combined_df = pd.concat(samples, ignore_index=True)
    deduped_index = combined_df.astype(str).drop_duplicates(keep="first").index
    deduped_configs = combined_df.iloc[deduped_index]
    
    return list(deduped_configs.to_dict("index").values())
