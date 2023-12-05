'''
The UserSim class is the method we use to simulate a user. 

They have the following fields: 
'''

class UserSim:
    def __init__(self):
        pass

    def get_response(self, query):
        print(query)
        response = input("Enter Response: ")
        return response