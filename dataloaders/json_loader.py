from dataloaders.dataloader import DataLoader
import json


class JSONLoader(DataLoader):

    '''
    Load the json data file into self.data
    '''
    def __init__(self, path, config): 
        super().__init__(config)
        with open(path) as json_data:
            jdata = json.load(json_data)
        self.data = jdata

    def __len__(self):
        return len(self.data)

    def get_data(self) -> dict:
        return self.data
    