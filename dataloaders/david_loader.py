from dataloaders.dataloader import DataLoader
import json


class DavidLoader(DataLoader):

    def __init__(self, config): #TODO: Fix for JSON
        super().__init__(config)
        with open(self.config['data']['data_path']) as json_data:
            jdata = json.load(json_data)
        self.data = jdata['items']

    def get_data(self) -> list:
        return self.data
    