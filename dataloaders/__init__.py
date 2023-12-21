from dataloaders.dataloader import DataLoader
from dataloaders.json_loader import JSONLoader
from dataloaders.user_tsv_loader import UserTsvLoader

# Used to create an instance of the appropriate data loader class based on the name provided in config.

DATALOADER_CLASSES = {
    'DataLoader': DataLoader,
    'JSONLoader': JSONLoader,
    'UserTsvLoader': UserTsvLoader,
}