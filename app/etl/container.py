from ssense_logger.app_logger import AppLogger

from app.config import Config
from app.library.scoring.scoring import Scoring
from app.etl.services.product_information_service import ProductInformationService
from app.etl.services.customer_interaction_service import CustomerInteractionService
from app.library.predict_data_import.remote_data_store.athena_data_store import \
    AwsAthenaConfig, CustomerInteractionDataStore, ProductInformationDataStore


class Container:

    def __init__(self, config: [Config] = Config):
        self.config = config
        self.app_logger = AppLogger(app_name=config.APP_NAME, env=config.ENV)
        self.customer_interaction_service = CustomerInteractionService(
            customer_interaction_source=CustomerInteractionDataStore(aws_config=AwsAthenaConfig(Config)),
            number_of_weeks_to_import=Config.LAST_N_WEEKS,
            scoring=Scoring(config.LAST_N_WEEKS,
                            config.P_WEIGHT,
                            config.W_WEIGHT,
                            config.DECAY_WEIGHT,
                            config.DECAY_WEIGHT_MULTIPLIER),
            logger=self.app_logger
        )
        self.product_information_service = ProductInformationService(
            product_information_source=ProductInformationDataStore(aws_config=AwsAthenaConfig(Config)),
            logger=self.app_logger
        )
