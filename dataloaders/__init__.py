from dataloaders.dataloader import DataLoader
from dataloaders.david_loader import DavidLoader
from dataloaders.rest_loader import RestaurantLoader

# Used to create an instance of the appropriate data loader class based on the name provided in config.

DATALOADER_CLASSES = {
    'DataLoader': DataLoader,
    'DavidLoader': DavidLoader,
    'RestLoader': RestaurantLoader
}