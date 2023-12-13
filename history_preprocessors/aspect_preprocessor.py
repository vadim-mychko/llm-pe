from history_preprocessors.history_preprocessor import HistoryPreprocessor

'''
The AspectPreprocessor converts the user's preferences into a string where the
aspects are concatenated. Aspects that the user's responded NO about have NOT 
in front of them
'''
class AspectPreprocessor(HistoryPreprocessor):

    def preprocess(self,history) -> str:
	#history = [{'query': query1, 'response': response1, 'aspect': aspect1, 'value': value1}, 
    #           {'query': query2, 'response': response2, 'aspect': aspect2, 'value': value2}]

    # Initialize an empty string to hold the concatenated history
        aspect_history = ""

    # Iterate over each interaction in the history
        for interaction in history:
            # strip and lowercase the response to see if it is "no"
            response = interaction['response'].strip().lower()
            contradiction = ""
            if response == "no":
                contradiction = "not "

            # Create aspect history
            concat_str = "%s: %s%s " % (interaction['aspect_key'], contradiction, interaction['aspect_value'])
            aspect_history += concat_str

        return aspect_history