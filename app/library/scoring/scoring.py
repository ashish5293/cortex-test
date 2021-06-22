import pandas as pd
import numpy as np
from pandas import DataFrame
from ssense_logger.app_logger import AppLogger

from app.config import Config

app_logger = AppLogger(app_name=Config.APP_NAME, env=Config.ENV)


class Scoring:
    """
    Class to calculate brand gender score based on user interaction history
    Scoring Config details:
        last_n_weeks: int # Number of weeks of data
        p_weight: Union[float, int]  # Weight used to score purchased product
        w_weight: Union[float, int]  # Weight used to score not purchased product added to wishlist or cart
        decay_weight: Union[float, int] # Coefficient to calculate decay
        decay_weight_multiplier: Union[float, int] # Coefficient to calculate decay
    """

    SCORING_TAGS = [Config.APP_NAME, 'scoring']

    def __init__(self,
                 last_n_weeks: int,
                 p_weight: float,
                 w_weight: float,
                 decay_weight: float,
                 decay_weight_multiplier: float):
        self.last_n_weeks = last_n_weeks
        self.p_weight = p_weight
        self.w_weight = w_weight
        self.decay_weight = decay_weight
        self.decay_weight_multiplier = decay_weight_multiplier

    def score_interactions(self, interactions: DataFrame) -> DataFrame:
        """
            Calculate brand gender score from a list of CustomerInteraction
            - weight brands purchased and added to the wishlist
            - apply time decay to the score
            - aggregate interactions, group by customer_id and sum score
            :param: DataFrame:interactions
            ['product_id', 'date', 'brand_id', 'gender', 'views', 'purchased',
            'add_to_cart', 'add_to_wishlist', 'time_on_page']
            :return: DataFrame: ['memberID', 'b_g', 'total_hits']
        """
        if interactions.size < 1:
            raise Exception('Can not score empty user interactions.')

        interactions = self._weight_interactions(interactions)
        interactions = self._apply_decay(interactions)
        interactions = self._aggregate_interactions(interactions)
        return interactions

    def _weight_interactions(self, interactions: DataFrame) -> DataFrame:
        """
            Prepare the user_interactions and weight the users-items interactions
            :param: DataFrame user_interactions
            :return: DataFrame: customer_interactions
        """
        app_logger.info(msg="Weighting customer interactions...", tags=self.SCORING_TAGS)

        # Combine brand and gender into one column
        interactions['b_g'] = interactions \
            .apply(lambda x: str(x.brand_id) + ' ' + str(x.gender), axis=1)

        # set views value for product purchased (purchased==1)
        interactions.loc[interactions.purchased == 1, 'views'] = \
            self.p_weight * interactions[interactions.purchased == 1]['views']

        # set views value for product added to wishlist or cart but not purchased  (purchased!=1)
        interactions.loc[((interactions.add_to_cart == 1)
                          | (interactions.add_to_wishlist == 1))
                         & (interactions.purchased != 1), 'views'] = \
            self.w_weight * interactions[((interactions.add_to_cart == 1)
                                          | (interactions.add_to_wishlist == 1))
                                         & (interactions.purchased != 1)]['views']
        return interactions

    def _get_decay_rate(self) -> float:
        """
        Decay formula
        """
        decay_weight = self.decay_weight
        decay_weight_multiplier = self.decay_weight_multiplier
        last_n_weeks = self.last_n_weeks
        return 1 / (decay_weight * decay_weight_multiplier * last_n_weeks)

    def _apply_decay(self, interactions: DataFrame) -> DataFrame:
        """
            Apply time decay function to the users-items interactions
            :param: DataFrame user_interactions
            :return: DataFrame: customer_interactions
        """
        app_logger.info(msg="Applying decay function to customer interactions...", tags=self.SCORING_TAGS)

        # Convert string date to datetime
        interactions.date = pd.to_datetime(interactions.date)
        # add a new column decay_date on the data frame with the decay function
        last_browsing_date = interactions.date.max()
        interactions = interactions.assign(decay_date=lambda x: (last_browsing_date - x.date)
                                           .astype('timedelta64[D]')
                                           .astype('int'))
        # add a new column decay calculated from decay_date and views
        decay_rate = self._get_decay_rate()
        interactions = interactions.assign(
            decay=lambda x: x.views * np.exp(-decay_rate * x.decay_date))
        return interactions

    def _aggregate_interactions(self, interactions: DataFrame) -> DataFrame:
        """
            Aggregate the users-items interactions, group the dataset by customer_id,
            brand, gender and sum the decay and cleanup the DataFrame
            :param: DataFrame user_interactions
            :return: DataFrame: customer_interactions
        """
        app_logger.info(msg="Aggregating customer interactions...", tags=self.SCORING_TAGS)

        interactions = interactions.groupby(['customer_id', 'b_g']) \
            .decay.sum() \
            .rename('views') \
            .reset_index()

        # remove unnecessary column
        interactions = interactions[['customer_id', 'b_g', 'views']]

        # rename columns to match model expectation
        interactions = interactions.rename(columns={'customer_id': 'memberID'})
        interactions = interactions.rename(columns={'views': 'total_hits'})
        return interactions
