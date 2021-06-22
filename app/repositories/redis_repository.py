from typing import List
import pandas as pd
from ssense_logger.app_logger import AppLogger
import re
from redis import StrictRedis
import time
import json

from app.config import Config


class RedisRepository:

    _WEEK_IN_SECONDS = 60 * 60 * 24 * 7
    _REDIS_PIPE_SIZE = 1000
    _REDIS_LPUSH_BLOCK_SIZE = 1000

    def __init__(self, config: Config, app_logger: AppLogger):
        self.redis = StrictRedis(host=config.REDIS_HOST,
                                 port=config.REDIS_PORT,
                                 encoding="utf-8",
                                 decode_responses=True)
        self.app_logger = app_logger

    def batch_save(self, data: pd.DataFrame):
        start_insert_time = time.time()
        grouped = data.groupby(by='memberID').apply(lambda x: x.to_dict(orient='records')).to_dict()

        pipe = self.redis.pipeline()
        # write _REDIS_PIPE_SIZE users in a row
        index = 0
        for key, value in grouped.items():
            for v in value:
                del v['memberID']
            pipe.set(f'scores:u:{key}', json.dumps(value), ex=self._WEEK_IN_SECONDS)
            index += 1
            if index > 0 and index % self._REDIS_PIPE_SIZE == 0:
                pipe.execute()
                pipe = self.redis.pipeline()
                self.app_logger.debug(f'---- inserted {index} record - {time.time() - start_insert_time}')

        # flush the pipe
        pipe.execute()
        self.app_logger.debug(f'---- inserted all {len(grouped)} records - {time.time() - start_insert_time}')

    def get_by_member_id(self, member_id: int) -> pd.DataFrame:
        """
        Gets all customer interactions of given customer
        :return: an array of CustomerInteractions
        """

        user_data_json = self.redis.get(f'scores:u:{member_id}')
        if user_data_json is not None:
            user_data = pd.read_json(user_data_json)
            return user_data

        return pd.DataFrame()

    def get_all_member_ids(self) -> List[int]:
        """
        :return: an array of integers containing all the available member ids with available user interactions
        """
        member_ids = []
        for key in self.redis.scan_iter("scores:u:*"):
            member_ids.append(int(re.sub('scores:u:', '', key)))
        return member_ids

    def push_member_ids_to_batch(self, member_ids: List[int]):
        """
        Creates the batch predict list if it doesnt exist
        Fills the batch predict list with member ids by blocks of size self._REDIS_LPUSH_BLOCK_SIZE
        :return: Nothing
        """
        step = self._REDIS_LPUSH_BLOCK_SIZE
        blocks = [member_ids[i * step:(i + 1) * step] for i in range((len(member_ids) + step - 1) // step)]
        for block in blocks:
            self.redis.lpush('members_to_predict_for', *block)

    def get_batch_customer_ids_to_process(self) -> List[int]:
        """
        Pull ids from the batch predict list in a blocking way so that multiple clients can work in parallel
        Note that by calling this method the ids returned are removed from the list
        List has a length of 1
        :return: List of customer ids
        """
        [list_name, member_id] = self.redis.blpop('members_to_predict_for')

        if member_id is None:
            return []

        return [int(member_id)]
