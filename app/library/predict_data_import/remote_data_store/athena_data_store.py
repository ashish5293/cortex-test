import boto3
import string
import time
from typing import Dict

from app.config import ConfigDBUpdateApp
from app.library.predict_data_import.remote_data_store.remote_data_store \
    import RemoteDataStore


class AwsAthenaConfig:
    def __init__(self, config: [ConfigDBUpdateApp]):
        self.aws_region = config.AWS_REGION
        self.aws_access_key_id = config.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = config.AWS_SECRET_ACCESS_KEY
        self.athena_db_name = config.ATHENA_DB_NAME
        self.athena_query_status_timeout = config.ATHENA_QUERY_STATUS_TIMEOUT
        self.athena_s3_output_location = 's3://' + config.ATHENA_S3_OUTPUT_BUCKET
        self.s3_bucket = config.ATHENA_S3_OUTPUT_BUCKET


class AthenaDataStore(RemoteDataStore):

    def __init__(self, aws_config: AwsAthenaConfig):
        self._aws_config = aws_config
        # Initialize athena client
        self._client = boto3.client(
            service_name='athena', region_name=aws_config.aws_region,
            aws_access_key_id=aws_config.aws_access_key_id, aws_secret_access_key=aws_config.aws_secret_access_key
        )
        # Initialize s3 client
        self.s3_client = boto3.resource(
            service_name='s3',
            region_name=aws_config.aws_region,
            aws_access_key_id=aws_config.aws_access_key_id,
            aws_secret_access_key=aws_config.aws_secret_access_key
        )

    def download_to_csv(self, local_file_path: str, number_of_week: int = None) -> None:
        """Download the query results to a local CSV file"""
        pass

    def query_and_download_result_csv(self, sql: str, local_file_path: str):
        # Execute query on athena
        query_id = self.start_query_execution(sql)
        self.wait_query_to_be_done(query_id)

        # Download CSV from S3 into local file
        self.s3_client.Bucket(self._aws_config.s3_bucket).download_file(
            query_id + '.csv',
            local_file_path)

    def get_query_status(self, query_id: str) -> Dict:
        status = self._client.get_query_execution(QueryExecutionId=query_id)['QueryExecution']['Status']
        if status['State'] in ['FAILED', 'CANCELLED']:
            raise RuntimeError(f'Athena query failed: {status}')
        return {
            'query_id': query_id,
            'is_done': status['State'] == 'SUCCEEDED'
        }

    def wait_query_to_be_done(self, query_id: string) -> None:
        seconds = 0
        while seconds < self._aws_config.athena_query_status_timeout:
            time.sleep(1)
            seconds += 1
            result = self.get_query_status(query_id)
            if result['is_done']:
                return
        # Something went wrong if we need more than
        # self._aws_config.athena_query_status_timeout seconds to verify query
        raise Exception(
            'No result when retrieving athena query status within {0}s.'.format(
                self._aws_config.athena_query_status_timeout))

    def start_query_execution(self, query: string) -> string:
        result = self._client.start_query_execution(
            QueryString=query, ResultConfiguration={
                'OutputLocation': self._aws_config.athena_s3_output_location
            },
            QueryExecutionContext={
                'Database': self._aws_config.athena_db_name
            }
        )
        return result['QueryExecutionId']


class CustomerInteractionDataStore(AthenaDataStore):

    def __init__(self, aws_config: AwsAthenaConfig = ConfigDBUpdateApp):
        super().__init__(aws_config)

    def download_to_csv(self, local_file_path: str, number_of_week: int = None) -> None:
        """
        This method download the user interaction data into a local CSV
        The query define the structure of the CSV file

        @:param local_file_path: str; file path to save data
        @:param: number_of_week: int; number of week to import
        @:return: None
        """
        query = "SELECT " \
                "ru.member_id as customer_id, " \
                "ru.product_id, " \
                "date_format(DATE(ru.date), '%Y-%m-%d') as date, " \
                "rp.brand_id as brand_id, " \
                "CASE rp.gender_t WHEN 'Men' THEN 1 WHEN 'Women' THEN 0 ELSE 2 END as gender, " \
                "ru.total_hits as views, " \
                "ru.purchased as purchased, " \
                "ru.add_to_cart, " \
                "ru.add_to_wishlist, " \
                "ru.total_on_page as time_on_page " \
                "FROM rec_user ru " \
                "JOIN rec_product rp ON ru.product_id=rp.product_id " \
                f"WHERE ru.date >= date_add('week', -{int(number_of_week)}, CURRENT_DATE) " \
                "AND ru.flag_reseller = 0"

        self.query_and_download_result_csv(query, local_file_path)


class ProductInformationDataStore(AthenaDataStore):

    def __init__(self, aws_config: AwsAthenaConfig = ConfigDBUpdateApp):
        super().__init__(aws_config=aws_config)

    def download_to_csv(self, local_file_path: str, number_of_week: int = None) -> None:
        """
        This method download the product information data into a local CSV
        The query define the structure of the CSV file

        @:param local_file_path: str; file path to save data
        @:param: number_of_week: int; number of week to import
        @:return: None
        """
        query = "SELECT " \
                "rp.product_id as productID, " \
                "rp.gender_t as gender, " \
                "rp.brand_id as brandID, " \
                "rp.seo_keyword as brand_seo, " \
                "rp.name as name, " \
                "rp.composition as composition, " \
                "rp.prod_creation_date as prodCreationDate, " \
                "rp.price_cd as priceCD, " \
                "rp.description as description, " \
                "rp.category as category, " \
                "rp.subcategory as subcategory, " \
                "rs.stock_for_sale as stockForSale " \
                "FROM rec_product rp " \
                "JOIN rec_stock rs ON rp.product_id=rs.product_id "

        self.query_and_download_result_csv(query, local_file_path)
