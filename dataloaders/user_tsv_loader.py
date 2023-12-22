from dataloaders.dataloader import DataLoader
import pandas as pd


class UserTsvLoader(DataLoader):

    '''
    Load the json data file into self.data
    '''
    def __init__(self, path, config): 
        super().__init__(config)
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        self.data = {}
        for row_num in range(len(df)):
            user_id = str(df.loc[row_num, 0])
            item_id = str(df.loc[row_num, 2])
            if not (user_id in self.data):
                self.data[user_id] = [item_id]
            else:
                self.data[user_id].append(item_id)

    def __len__(self):
        return len(self.data)

    def get_data(self) -> dict:
        return self.data
    