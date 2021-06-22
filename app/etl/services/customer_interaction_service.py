import os
import time
import pandas as pd
from ssense_logger.app_logger import AppLogger

from app.library.scoring.scoring import Scoring
from app.etl.services.base_etl_service import BaseEtlService
from app.library.predict_data_import.remote_data_store.athena_data_store \
    import CustomerInteractionDataStore


class CustomerInteractionService(BaseEtlService):
    _SERVICE_TAG = 'customer_interaction_service'
    _DOWNLOAD_DIR = 'csv_download'
    _LOCAL_FILE_PATH = _DOWNLOAD_DIR + '/customer_interaction_dump.csv'

    def __init__(self,
                 customer_interaction_source: CustomerInteractionDataStore,
                 number_of_weeks_to_import: int,
                 scoring: Scoring,
                 logger: AppLogger):
        self.customer_interaction_source = customer_interaction_source
        self.number_of_weeks_to_import = number_of_weeks_to_import
        self.scoring = scoring
        self.logger = logger

    def extract(self) -> None:
        """
        Fetch customer interaction data and extract
        the result into a local CSV file.

        :return: None
        """
        if not os.path.exists(self._DOWNLOAD_DIR):
            os.makedirs(self._DOWNLOAD_DIR)
        try:
            self.logger.info(msg='Start customer interactions extract...',
                             tags=[self._SERVICE_TAG, 'extract', 'start'])

            start_time = time.time()
            self.customer_interaction_source.download_to_csv(self._LOCAL_FILE_PATH, self.number_of_weeks_to_import)
            self.logger.info(msg=f'Customer interactions extract done. Total elapsed time: '
                                 f'{round(time.time() - start_time, 2)} seconds',
                             tags=[self._SERVICE_TAG, 'extract', 'done'])
        except Exception as e:
            self.logger.error(msg=f'Something went wrong when downloading '
                                  f'user interactions into: {self._LOCAL_FILE_PATH}',
                              tags=[self._SERVICE_TAG, 'extract', 'csv', 'error'])
            raise Exception(f'Something went wrong when proceeding'
                            f' to customer interactions extraction: {e}')

    def transform(self, customer_interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Proceed to customer interaction data scoring

        :param customer_interactions: pd.DataFrame
        :return: pd.DataFrame
        """
        try:
            self.logger.info(msg='Start customer interactions scoring...',
                             tags=[self._SERVICE_TAG, 'scoring', 'start'])

            start_time = time.time()
            customer_interactions = self.scoring.score_interactions(customer_interactions)

            self.logger.info(msg=f'Customer interaction scoring done. Total elapsed time: '
                                 f'{round(time.time() - start_time, 2)} seconds',
                             tags=[self._SERVICE_TAG, 'scoring', 'done'])
            return customer_interactions
        except Exception as e:
            self.logger.error(msg=f'Something went wrong when proceeding'
                                  f' to customer interactions scoring: {e}',
                              tags=[self._SERVICE_TAG, 'scoring', 'error'])
            raise Exception(f'Something went wrong when proceeding'
                            f' to customer interactions scoring: {e}')

    def load(self) -> pd.DataFrame:
        """
        Load the customer interaction data from
        the csv into a DataFrame

        :return: customer_interactions: ps.DataFrame
        """
        try:
            return pd.read_csv(self._LOCAL_FILE_PATH,
                               index_col=0,
                               encoding='utf-8')
        except Exception as e:
            self.logger.error(msg=f'Something went wrong when loading data from csv file: {self._LOCAL_FILE_PATH}',
                              tags=[self._SERVICE_TAG, 'load', 'csv', 'error'])
            raise Exception(f'Something went wrong when proceeding'
                            f' to customer interactions loading: {e}')
