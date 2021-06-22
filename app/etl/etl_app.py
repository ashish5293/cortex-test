import time
from typing import Tuple
from ssense_logger.app_logger import AppLogger

from app.etl.container import Container


class EtlApp:

    def __init__(self, container: Container, app_logger: AppLogger):
        self.container = container
        self.customer_interaction_service = container.customer_interaction_service
        self.product_information_service = container.product_information_service
        self.app_logger = app_logger

    def run(self) -> Tuple:
        start_time = time.time()

        try:
            self.app_logger.info(msg='Starting ETL...', tags=['etl_app', 'start'])
            self.customer_interaction_service.extract()
            hits_dataset = self.customer_interaction_service.load()
            hits_dataset = self.customer_interaction_service.transform(hits_dataset)
            self.product_information_service.extract()
            products_dataset = self.product_information_service.load()
            products_dataset = self.product_information_service.transform(products_dataset, 'brandID', 'gender')

            self.app_logger.info(msg='ETL app completed! '
                                     f'total time: {round(time.time() - start_time, 2)} sec.',
                                 tags=['etl_app', 'completed'])
            return hits_dataset, products_dataset
        except Exception as e:
            self.app_logger.error(msg=f'ETL app failed with error {e}',
                                  tags=['etl_app', 'error'])
            raise e
