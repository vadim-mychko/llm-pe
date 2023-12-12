from item_scorers.item_scorer import ItemScorer
import llms
import math
import jinja2

class LLMLogprobScorer(ItemScorer):

    def __init__(self, config):
        super().__init__(config)  

        llm_module = llms.LLM_CLASSES[self.config['llm']['llm_name']]
        self.llm = llm_module(config)
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=self.config['llm']['template_dir']))


    # Check if the user will like an item based on the item description and the full/partial interaction history
    # Return the probability that the user will like the item as a float.
    def score_items(self,preference,items) -> dict:
        #Anton Dec 11: changed variable name interaction_history to preference to account for possible transformations of interaction_history (e.g., extracting aspects user likes, etc - particularly relevant to entailment models)
        like_probs = {item_id : None for item_id in items}
        template_file = self.config['llm']['like_probs_template']
        query_template = self.jinja_env.get_template(template_file)

        for item_id in items:

            #probability of liking item
            like_prob = 0.0

            context = {
                "item_desc": items[item_id]['description'],
                "interactions": preference
            }
            query = query_template.render(context)

            # self.logger.debug(query)

            # response = self.llm.make_request(query, logprobs=0)
            response = self.llm.make_request(query, logprobs=1)

            response = response.strip().lower()

            full_logprobs = self.llm.get_full_logprobs() 

            # NOTE: We're assuming here that response is one word, which it should be. Take logprobs of the first word

            log_probs = full_logprobs['token_logprobs'][0]

            probs = math.exp(log_probs)

            if (response == "true"):
                like_prob = (1 + probs) / 2
            elif (response == "false"):
                like_prob = (1 - probs) / 2
            else:
                raise ValueError("Expected LLM response to be either true or false")
        
            # self.logger.debug("Like Probs Query: %s Response: %s Like Probs: %f" % (query, response, like_probs))
        
            like_probs[item_id] = like_prob

        return like_probs
