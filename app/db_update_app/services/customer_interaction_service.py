import os
import time
import pandas as pd
from ssense_logger.app_logger import AppLogger

from app.library.scoring.scoring import Scoring
from app.repositories.redis_repository import RedisRepository
from app.db_update_app.services.base_update_service import BaseUpdateService
from app.library.predict_data_import.remote_data_store.athena_data_store \
    import CustomerInteractionDataStore


class CustomerInteractionService(BaseUpdateService):
    _SERVICE_TAG = 'customer_interaction_service'
    _DOWNLOAD_DIR = 'csv_download'
    _LOCAL_FILE_PATH = _DOWNLOAD_DIR + '/customer_interaction_dump.csv'

    def __init__(self,
                 customer_interaction_source: CustomerInteractionDataStore,
                 local_source: RedisRepository,
                 number_of_weeks_to_import: int,
                 min_number_of_record_expected: int,
                 scoring: Scoring,
                 logger: AppLogger):
        self.data_remote_source = customer_interaction_source
        self.min_record_expected = min_number_of_record_expected
        self.number_of_weeks_to_import = number_of_weeks_to_import
        self.local_source = local_source
        self.scoring = scoring
        self.logger = logger

    def update(self) -> None:
        """
        Update local database (REDIS) with
        new customer interaction data.

        :return: None
        """

        self._download_customer_interactions_into_csv()
        customer_interactions = self._load_customer_interactions_from_csv()

        self._verify_min_record_expected(c_interactions=customer_interactions,
                                         n_record_expected=self.min_record_expected)

        scored_interactions = self._score_customer_interactions(customer_interactions)
        self._insert_into_local_source(scored_interactions)

    def _download_customer_interactions_into_csv(self) -> None:
        """
        Fetch the last N week of data (customer interactions) from
        remote data source (Athena) and download it into a CSV file

        :return: None
        """
        start_time = time.time()
        self.logger.info(msg=f'Downloading last {self.number_of_weeks_to_import} weeks of data into '
                             f'local CSV: {self._LOCAL_FILE_PATH} ...',
                         tags=[self._SERVICE_TAG])

        if not os.path.exists(self._DOWNLOAD_DIR):
            os.makedirs(self._DOWNLOAD_DIR)

        # Download data from remote source
        self.data_remote_source.download_to_csv(self._LOCAL_FILE_PATH,
                                                self.number_of_weeks_to_import)
        read_time = time.time()
        try:
            file_size = os.path.getsize(self._LOCAL_FILE_PATH)
        except OSError:
            # we don't block the import on this error
            file_size = '~'

        self.logger.info(msg=f"CSV file ({file_size} bytes) downloaded in: "
                             f"{round(read_time - start_time, 2)} sec.",
                         tags=[self._SERVICE_TAG])

    def _load_customer_interactions_from_csv(self):
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
            return pd.DataFrame()

    def _verify_min_record_expected(self,
                                    c_interactions: pd.DataFrame,
                                    n_record_expected: int) -> None:
        """
        Verify that data downloaded from Athena is enough
        and not corrupted checking the count of record
        against {min_records_expected}

        :param c_interactions: pd.DataFrame
        :param n_record_expected: int
        :return: None
        """
        fetched_records_count = c_interactions.size

        if fetched_records_count <= n_record_expected:
            self.logger.error(msg=f'Something went wrong when downloading {fetched_records_count} from athena. '
                                  f'Expected at least {n_record_expected}',
                              tags=[self._SERVICE_TAG, 'error'])
            raise Exception(f'Downloaded approximately {fetched_records_count} from athena. '
                            f'Expected at least {n_record_expected}')
        else:
            self.logger.info(msg=f'{fetched_records_count} downloaded record check done.',
                             tags=[self._SERVICE_TAG, 'data', 'verified'])

    def _score_customer_interactions(self, customer_interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Proceed to customer interactions scoring

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

    def _insert_into_local_source(self, customer_interactions: pd.DataFrame) -> None:
        """
        Populate local source (REDIS) with
        the scored customer interactions

        :param customer_interactions: pd.DataFrame
        :return: None
        """
        try:
            self.logger.info(msg='Start populating redis...', tags=[self._SERVICE_TAG])

            start_time = time.time()
            self.local_source.batch_save(customer_interactions)

            self.logger.info(msg=f' Insertion into Redis done. Total elapsed time: '
                                 f'{round(time.time() - start_time, 2)} seconds',
                             tags=[self._SERVICE_TAG, 'redis', 'done'])
        except Exception as e:
            self.logger.error(msg=f'Something went wrong when proceeding'
                                  f' to populating Redis: {e}',
                              tags=[self._SERVICE_TAG, 'redis', 'error'])
            raise Exception(f'Something went wrong when proceeding'
                            f' to populating Redis: {e}')
