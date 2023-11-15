from dataloaders.dataloader import DataLoader
import json


class RestaurantLoader(DataLoader):

    '''
    Load the json data file into self.data
    '''
    def __init__(self, config): 
        super().__init__(config)
        with open(self.config['data']['data_path']) as json_data:
            jdata = json.load(json_data)

        self.data = jdata

    def __len__(self):
        return len(self.data)

    def get_data(self, k=3) -> list:
        # TODO: Return the dict with the top k reviews for each restaurant
        return_list = []
        id_counter = 0
        for key in self.data.keys():
            num_revs = k
            if (num_revs >= len(self.data[key])):
                num_revs = len(self.data[key])
             
            reviews = self.data[key][:num_revs]
            item = {'id': id_counter, 'name': key, 'reviews': reviews}
            id_counter += 1
            return_list.append(item)

        return return_list
    