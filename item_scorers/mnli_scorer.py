import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from item_scorers.item_scorer import ItemScorer

class MNLIScorer(ItemScorer):

    def __init__(self, config):
        super().__init__(config)
        
        self.nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        self.nli_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

        import pdb; pdb. set_trace()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.nli_model.to(self.device)

    def score_items(self,preference,items) -> dict:

        hypotheses = [preference] * len(items)
    
        premises = [item['description'] for item in items.values()]

        inputs = self.nli_tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt", max_length=self.nli_tokenizer.model_max_length)

        # Ensure the tokenized inputs don't exceed the max length
        if inputs.input_ids.size(1) > self.nli_tokenizer.model_max_length:
            print(f"Warning: One or more tokenized inputs exceed the max length of {self.tokenizer.model_max_length} and will be truncated.")

        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        #get entailement probs
        probabilities = predictions[:, 2].tolist()
        entailment_probs = {item_id : None for item_id in items}

        for i, item_id in enumerate(items):
            entailment_probs[item_id] = probabilities[i]

        return entailment_probs