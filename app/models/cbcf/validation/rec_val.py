"""
@name: rec_val.py
@author: Mohammad Jeihoonian
Created on Jan 2020
"""
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.sparse import csr_matrix

from app.config import Config
from app.models.cbcf.training.cb_train import ContTrain
from app.models.cbcf.training.cf_train import CollabTrain
from app.utils.exception_decorator import exception_decorator
from app.models.cbcf.validation.rec_metrics import ValidationMetrics
from ssense_logger.app_logger import AppLogger

# Initialize Logger
app_logger = AppLogger(app_name=Config.APP_NAME, env=Config.ENV)


VAL_PARAMS = dict(n_val=Config.N_VAL,
                  k_val=Config.K_VAL,
                  alpha_val=Config.ALPHA_VAL,
                  strategy=Config.STRATEGY)

REFERENCE_DATA_PARAMS = dict(
    # These vars are used to collect the reference point data. It's not run often, but refreshed
    # every once in a while for sanity checks when training the models.
    bg_cutpoint=Config.BG_CUTPOINT,  # <- used by etl (used during the transformation)
    cycle_count=Config.CYCLE_COUNT,  # <- used by etl (used during the transformation)
)


class RecVal(object):

    def __init__(self, hits: pd.DataFrame, products: pd.DataFrame):

        self.hits_data = hits
        self.products_data = products
        self.n_rec = VAL_PARAMS['n_val']
        self.k_val = VAL_PARAMS['k_val']
        self.alpha_val = VAL_PARAMS['alpha_val']
        self.strategy = VAL_PARAMS['strategy']

    @staticmethod
    def _gender_processing(dataset):

        dataset = dataset.copy()
        dataset = dataset.assign(norm_score=lambda x: x.log_score / dataset.log_score.max())
        dataset.rename(columns={'norm_score': 'score'}, inplace=True)

        return dataset[['brand', 'gender', 'score', 'liked']]

    @staticmethod
    def train(class_obj):

        sim_mat, item_dict = class_obj.fit()

        return sim_mat, item_dict

    @staticmethod
    def _rescale_cb(dataset, ref_dataset) -> pd.DataFrame:

        def interp_func(x, lb):
            return ((x - min(x)) / (1 - min(x))) * (1 - lb) + lb

        dataset_out = pd.DataFrame()

        for gender in [0, 1]:

            dataset_sub = dataset[dataset.gender == gender].copy()

            if dataset_sub.empty:
                continue

            lower_bound = ref_dataset.groupby('gender')['score'].apply(min)[gender]

            dataset_sub = dataset_sub.assign(score=lambda x: interp_func(x['score'], lb=lower_bound))

            dataset_out = dataset_out.append(dataset_sub)

        return dataset_out.sort_values(by='score', ascending=False).reset_index(drop=True)

    @staticmethod
    def _val_prep(userId, act, pred):

        df = pd.DataFrame()

        for genderT in (0, 1):

            if not act[act.gender == genderT].empty:

                dataset = pd.DataFrame({'memberID': [userId]})

                dataset = dataset.assign(gender=genderT,
                                         test=np.empty((dataset.shape[0], 0)).tolist(),
                                         pred=np.empty((dataset.shape[0], 0)).tolist())

                dataset.at[0, 'test'] = list(act[act.gender == genderT].b_g.tolist())

                dataset.at[0, 'pred'] = list(pred[pred.gender == genderT].b_g.tolist())

                df = df.append(dataset)

        if not df.empty:

            return df

    def _brand_gender_split(self):

        self.hits_data['brand'], self.hits_data['gender'] = self.hits_data.b_g.str.split(' ').str
        self.hits_data.brand = self.hits_data.brand.astype('int16')
        self.hits_data.gender = self.hits_data.gender.astype('int8')

        return self.hits_data

    def _train_test_split(self):
        """
        Label the parameters as a cycle of labels
        :return: train and test sets
        """
        app_logger.info(msg=f'Incoming data has {self.hits_data.shape[0]} rows for {self.hits_data.memberID.nunique()} members')
        hits_sub = self.hits_data.copy()
        hits_sub = hits_sub.groupby(['memberID', 'gender']) \
            .size() \
            .reset_index(name='counts_gender')
        hits_sub = hits_sub[hits_sub['counts_gender'] >= REFERENCE_DATA_PARAMS['bg_cutpoint']]
        hits_sub.drop('counts_gender', axis=1, inplace=True)
        hits_sub.reset_index(inplace=True, drop=True)
        # rejoin on original df, dropping small member hit counts
        hits_sub = pd.merge(self.hits_data, hits_sub, how='inner', on=['memberID', 'gender'])
        # take modulo (rolling count of brand genders per member)
        hits_sub['count_roll'] = hits_sub.groupby(['memberID', 'gender']).cumcount() % REFERENCE_DATA_PARAMS[
            'cycle_count']
        random_row = np.random.randint(0, REFERENCE_DATA_PARAMS['cycle_count'])
        # assign random row within each cycle to be for out-of-sample validation test
        hits_sub['test_train'] = np.where(hits_sub['count_roll'] == random_row, 'test', 'train')
        app_logger.info(msg=f'Outgoing data has {hits_sub.shape[0]} rows for {hits_sub.memberID.nunique()} members')
        train_set = hits_sub[hits_sub.test_train == 'train'].copy()
        train_set.drop(['test_train', 'count_roll'], axis=1, inplace=True)
        app_logger.info(msg=f'The number of users in train set: {train_set["memberID"].nunique()}', tags=['check_data'])
        test_set = hits_sub[hits_sub.test_train == 'test'].copy()
        test_set.drop(['test_train', 'count_roll'], axis=1, inplace=True)
        app_logger.info(msg=f'The number of users in test set: {test_set["memberID"].nunique()}', tags=['check_data'])

        return train_set, test_set

    def _post_process_rec(self, dataset):
        """
        Process recommendations
        :param dataset
        :return dataset
        """

        dataset = dataset.assign(log_score=lambda x: np.log1p(x.score))
        dataset.drop(columns=['score'], axis=1, inplace=True)

        dataset.brand = dataset.brand.astype('int16')
        dataset.gender = dataset.gender.astype('int8')

        proc_dataset = pd.DataFrame()
        for genderT in (1, 0):
            dataset_gender = dataset[dataset.gender == genderT].head(self.n_rec).copy()
            dataset_gender = RecVal._gender_processing(dataset_gender)
            proc_dataset = proc_dataset.append(dataset_gender, sort=False)

        proc_dataset.sort_values(by='score', ascending=False, inplace=True)
        proc_dataset.reset_index(inplace=True, drop=True)

        proc_dataset.score = np.round(proc_dataset.score, decimals=6)

        return proc_dataset

    @exception_decorator
    def _rec_predict(self, user_data, sim_mat: csr_matrix, item_dict: dict):

        """
        :return recommendations for each user in the dataset
        """

        user_data_dict = dict(zip(user_data.b_g, user_data.total_hits))

        user_items = np.zeros(len(item_dict))
        for i in range(user_items.shape[0]):
            if item_dict[i] in list(user_data_dict):
                user_items[i] = user_data_dict[item_dict[i]]

        user_items = user_items.reshape(1, -1)
        user_items = csr_matrix(user_items)

        # Compute dot product
        rec_mat = user_items @ sim_mat

        result = []
        liked = set(user_items.indices)
        user_indices, user_scores = rec_mat.indices, rec_mat.data
        best = sorted(zip(user_indices, user_scores), key=lambda x: -x[1])
        tagged_best = [rec + (False,) for rec in best if rec[0] not in liked]
        result.extend([(item_dict[rid].split(' ')[0], item_dict[rid].split(' ')[1],
                        score, flag_brx) for rid, score, flag_brx in tagged_best])

        rec = pd.DataFrame(result, columns=['brand', 'gender', 'score', 'liked'])

        if rec.score.lt(0).any():
            return pd.DataFrame(columns=['brand', 'gender', 'score', 'liked'])

        return self._post_process_rec(rec)

    def predict(self, user_train_hits, sim_mat, item_dict):

        rec_df = self._rec_predict(user_train_hits, sim_mat, item_dict)

        rec_df = rec_df.assign(b_g=lambda x: x.brand.astype('str') + ' ' + x.gender.astype('str'))

        return rec_df

    def batch_predict(self, users_array, train_hits, test_hits, sim_mat, item_dict):

        pred_df = pd.DataFrame()
        val_df = pd.DataFrame()

        for userId in users_array:

            user_train_hits = train_hits[train_hits.memberID == userId].copy()

            pred = self.predict(user_train_hits, sim_mat, item_dict)

            pred = pred.assign(memberID=userId)

            pred_df = pred_df.append(pred)

            val = RecVal._val_prep(userId, test_hits[test_hits.memberID == userId], pred)

            val_df = val_df.append(val)

        val_df.reset_index(inplace=True, drop=True)

        return pred_df, val_df

    def _rec_agg(self, cf_rec: pd.DataFrame, cb_rec: pd.DataFrame, test_hits: pd.DataFrame) -> Tuple:

        cbf = cf_rec.set_index(['memberID', 'brand', 'gender'])\
            .join(cb_rec.set_index(['memberID', 'brand', 'gender']),
                  how='outer', lsuffix='_cf', rsuffix='_cb')

        cbf.reset_index(inplace=True)
        cbf.liked_cf.fillna(cbf.liked_cb, inplace=True)
        cbf.liked_cb.fillna(cbf.liked_cf, inplace=True)
        cbf = cbf.assign(score=lambda x: np.where(x.score_cf.isnull(),
                                                  x.score_cb,
                                                  np.where(
                                                      x.score_cb.isnull(),
                                                      x.score_cf,
                                                      ((self.alpha_val * x.score_cb) + ((1 - self.alpha_val) * x.score_cf)))),
                         liked=lambda x: np.where(x.liked_cf == x.liked_cb, x.liked_cf, x.liked_cf)
                         )
        cbf = cbf.sort_values(by=['memberID', 'score'], ascending=[True, False])
        cbf.reset_index(drop=True, inplace=True)

        cbf = cbf[['memberID', 'brand', 'gender', 'score', 'liked']]
        cbf = cbf.assign(b_g=lambda x: x.brand.astype('str') + ' ' + x.gender.astype('str'))

        val_df = pd.DataFrame()

        for userId in cbf.memberID.unique():

            val = RecVal._val_prep(userId, test_hits[test_hits.memberID == userId], cbf[cbf.memberID == userId])

            val_df = val_df.append(val)

        val_df.reset_index(inplace=True, drop=True)

        return cbf, val_df

    def _val_score_gender(self, dataset: pd.DataFrame):

        if not dataset.empty:

            gender_dict = {0: 'women', 1: 'men'}

            for genderT in list(gender_dict):

                dataset_gender = dataset[dataset.gender == genderT].copy()

                if not dataset_gender.empty:

                    metrics = ValidationMetrics(actual=dataset_gender.test.tolist(),
                                                predict=dataset_gender.pred.tolist(),
                                                k_val=self.k_val)

                    return metrics.grouping_score_ranked()

    def model_val(self):

        self.hits_data = self._brand_gender_split()

        train_hits, test_hits = self._train_test_split()

        users_array = np.random.choice(train_hits.memberID.unique(), size=10000, replace=False)

        cf_sim_mat, cf_item_dict = self.train(CollabTrain(train_hits))
        cf_df, cf_val_df = self.batch_predict(users_array, train_hits, test_hits, cf_sim_mat, cf_item_dict)

        self._val_score_gender(cf_val_df)
        cb_sim_mat, cb_item_dict = self.train(ContTrain(self.products_data))
        cb_df, cb_val_df = self.batch_predict(users_array, train_hits, test_hits, cb_sim_mat, cb_item_dict)

        self._val_score_gender(cb_val_df)

        cb_df = RecVal._rescale_cb(cb_df, cf_df)

        cbf_df, cbf_val_df = self._rec_agg(cf_df, cb_df, test_hits)

        return self._val_score_gender(cbf_val_df)
