import time
from ssense_logger.app_logger import AppLogger

from app.db_update_app.container import Container
from app.helpers.alert_helper import AlertHelper


class DbUpdateApp:

    def __init__(self, container: Container, app_logger: AppLogger):
        self.container = container
        self.customer_interaction_service = container.services.customer_interaction_service
        self.alert_helper = AlertHelper.get_instance()
        self.app_logger = app_logger

    def start(self):
        start_time = time.time()

        try:
            self.app_logger.info(msg=f'{self.container.config.APP_NAME} starting...', tags=['db_update_app', 'start'])
            self.customer_interaction_service.update()
            self.app_logger.info(msg=f'{self.container.config.APP_NAME} finished! '
                                     f'total time: {round(time.time() - start_time, 2)} sec.',
                                 tags=['db_update_app', 'end'])

        except Exception as e:
            self.app_logger.error(msg=f'{self.container.config.APP_NAME} failed with error {e}',
                                  tags=['db_update_app', 'error'])
            self.alert_helper.error(f':sad-prateek: Db update app failed with error {e.__str__()} :sad-prateek:')
