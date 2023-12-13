from history_preprocessors.history_preprocessor import HistoryPreprocessor

class HistoryConcatenator(HistoryPreprocessor):

    def preprocess(self,history) -> str:
	#history = [{'query': query1, 'response': response1},{'query': query2, 'response': response2}]

    # Initialize an empty string to hold the concatenated history
        concatenated_history = ""

    # Iterate over each interaction in the history
        for interaction in history: # David Dec13: switching name from item to interaction, since it's confusing since items aren't involved
            # Append the 'query' and 'response' of each item to the string
            concatenated_history += "Query: " + interaction['query'] + " \n Response: " + interaction['response'] + " \n "

        return concatenated_history