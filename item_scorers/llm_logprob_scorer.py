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
        like_probs = {item_id : None for item_id in items}
        template_file = self.config['llm']['like_probs_template']
        query_template = self.jinja_env.get_template(template_file)

        for item_id in items:

            #probability of liking item
            like_prob = 0.0

            context = {
                "item_desc": items[item_id]['description'],
                "preference": preference
            }
            query = query_template.render(context)


            # self.logger.debug(query)

            # response = self.llm.make_request(query, logprobs=0)
            response = self.llm.make_request(query, logprobs=2)

            try:

                logprobs = self.llm.get_logprobs()

                (k,v) = list(logprobs.items())[0]

                if k == 'Unc':
                        like_prob = 0.5
                elif k == 'False':
                        like_prob = 1 - math.exp(v)
                elif k == 'True':
                        like_prob = math.exp(v)


            except KeyError:
                print("Neither 'True' nor 'False' key found in the logprobs dictionary")

            # self.logger.debug("Like Probs Query: %s Response: %s Like Probs: %f" % (query, response, like_probs))
        
            like_probs[item_id] = like_prob

        return like_probs
