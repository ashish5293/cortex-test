from abc import ABC, abstractmethod


class RemoteDataStore(ABC):
    """
    This class is the interface that declare the methods required to interact with the remote repositories
    """
    @abstractmethod
    def download_to_csv(self, local_file_path: str, number_of_week: int = None) -> None:
        """Download the query results to local CSV file"""
        pass
