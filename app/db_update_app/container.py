from ssense_logger.app_logger import AppLogger

from app.config import ConfigDBUpdateApp
from app.db_update_app.services.customer_interaction_service import CustomerInteractionService
from app.helpers.alert_helper import AlertHelper
from app.library.predict_data_import.remote_data_store.athena_data_store \
    import CustomerInteractionDataStore, AwsAthenaConfig
from app.library.scoring.scoring import Scoring
from app.repositories.redis_repository import RedisRepository


class Container:

    def __init__(self, config: ConfigDBUpdateApp = ConfigDBUpdateApp):
        self.config = config

        AlertHelper(config)

        self.repositories = Repositories(config=self.config)
        self.services = Services(repositories=self.repositories, config=config)


class Repositories:
    def __init__(self, config: ConfigDBUpdateApp):
        app_logger = AppLogger(app_name=config.APP_NAME, env=config.ENV)

        self.remote_customer_interactions_source = CustomerInteractionDataStore(AwsAthenaConfig(config))
        self.local_source = RedisRepository(config=config, app_logger=app_logger)


class Services:
    def __init__(self, repositories: Repositories, config: ConfigDBUpdateApp):
        self.config = config
        self.scoring = Scoring(
            self.config.LAST_N_WEEKS,
            self.config.P_WEIGHT,
            self.config.W_WEIGHT,
            self.config.DECAY_WEIGHT,
            self.config.DECAY_WEIGHT_MULTIPLIER)

        self.app_logger = AppLogger(app_name=config.APP_NAME, env=config.ENV)

        self.customer_interaction_service = CustomerInteractionService(
            customer_interaction_source=repositories.remote_customer_interactions_source,
            local_source=repositories.local_source,
            number_of_weeks_to_import=config.NUMBER_OF_WEEKS_TO_IMPORT,
            min_number_of_record_expected=config.MIN_NUMBER_OF_RECORD_EXPECTED,
            scoring=self.scoring,
            logger=self.app_logger
        )
