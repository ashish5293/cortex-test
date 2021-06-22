import os
import time
import numpy as np
import pandas as pd
from ssense_logger.app_logger import AppLogger

from app.etl.services.base_etl_service import BaseEtlService
from app.library.predict_data_import.remote_data_store.athena_data_store \
    import ProductInformationDataStore


class ProductInformationService(BaseEtlService):
    _SERVICE_TAG = 'product_information_service'
    _DOWNLOAD_DIR = 'csv_download'
    _LOCAL_FILE_PATH = _DOWNLOAD_DIR + '/product_information_dump.csv'

    def __init__(self,
                 product_information_source: ProductInformationDataStore,
                 logger: AppLogger):
        self.product_information_source = product_information_source
        self.logger = logger

    def extract(self) -> None:
        """
        Fetch products information and extract
        the result into a local CSV file.

        :return: None
        """
        if not os.path.exists(self._DOWNLOAD_DIR):
            os.makedirs(self._DOWNLOAD_DIR)
        try:
            self.logger.info(msg='Start products information extract...',
                             tags=[self._SERVICE_TAG, 'extract', 'start'])

            start_time = time.time()
            self.product_information_source.download_to_csv(self._LOCAL_FILE_PATH)
            self.logger.info(msg=f'Products information extract done. Total elapsed time: '
                                 f'{round(time.time() - start_time, 2)} seconds',
                             tags=[self._SERVICE_TAG, 'extract', 'done'])
        except Exception as e:
            self.logger.error(msg=f'Something went wrong when downloading '
                                  f'products information into: {self._LOCAL_FILE_PATH}',
                              tags=[self._SERVICE_TAG, 'extract', 'csv', 'error'])
            raise Exception(f'Something went wrong when proceeding'
                            f' to products information extraction: {e}')

    def transform(self, products_dataset: pd.DataFrame, *args) -> pd.DataFrame:
        dataset = products_dataset.copy()

        for arg in args:
            if arg == 'brandID':
                dataset[arg] = dataset[arg].astype('str')
            elif arg == 'gender':
                dataset[arg] = dataset[arg].str.lower()
            else:
                dataset[arg] = dataset[arg].str.lower()
                dataset[arg] = dataset[arg].str.replace("-", "")
                dataset[arg] = dataset[arg].str.replace("_", "")
                dataset[arg] = dataset[arg].str.replace("&", "and")

        dataset = dataset.assign(
            genderID=lambda x: np.where(x.gender == 'women', '0',
                                        np.where(x.gender == 'men', '1', '2')
                                        ),
            b_g=lambda x: x.brandID + ' ' + x.genderID,
        )
        return dataset

    def load(self) -> pd.DataFrame:
        """
        Load the products information data from
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
                            f' to productions information loading: {e}')
