from history_preprocessors.history_preprocessor import HistoryPreprocessor

class HistoryConcatenator(HistoryPreprocessor):

    def preprocess(self,history) -> str:
	#history = [{'query': query1, 'response': response1},{'query': query2, 'response': response2}]

    # Initialize an empty string to hold the concatenated history
        concatenated_history = ""

    # Iterate over each item in the history
        for item in history:
            # Append the 'query' and 'response' of each item to the string
            concatenated_history += "Query: " + item['query'] + " \nResponse: " + item['response'] + " \n "

        return concatenated_history