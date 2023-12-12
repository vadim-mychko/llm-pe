
from item_scorers.item_scorer import ItemScorer
import torch
from sentence_transformers import CrossEncoder


class CEScorer(ItemScorer):

    def __init__(self, config):
        super().__init__(config)
        
        self.model = CrossEncoder(self.config['llm']['entailement_model'])

    def score_items(self,preference,items) -> dict:

        like_probs = {item_id : None for item_id in items}

        inputs = [(preference, items[item_id]['description']) for item_id in items]
        
        #get CE logits
        output = self.model.predict(inputs)

        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(torch.from_numpy(output))

        for i, item_id in enumerate(items):
            like_probs[item_id] = probabilities[i].item()

        return like_probs