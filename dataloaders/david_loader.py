from dataloaders.dataloader import DataLoader
import pandas as pd


class DavidLoader(DataLoader):

    def __init__(self, config):
        super().__init__(config)
        df = pd.read_csv(self.config['data']['data_path'])
        self.data = df

    def get_data(self) -> list:
        return self.data
    