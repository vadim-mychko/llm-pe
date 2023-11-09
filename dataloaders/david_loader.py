from dataloaders.dataloader import DataLoader
import json


class DavidLoader(DataLoader):

    '''
    Load the json data file into self.data
    '''
    def __init__(self, config): 
        super().__init__(config)
        with open(self.config['data']['data_path']) as json_data:
            jdata = json.load(json_data)
        self.data = jdata['items']

    def __len__(self):
        return len(self.data)

    def get_data(self) -> list:
        return self.data
    