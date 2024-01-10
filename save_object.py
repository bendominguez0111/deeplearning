import pickle

class Value:
    def __init__(self, data):
        self.data = data

v = Value(1)

file = open('file.obj', 'w')
pickle.dump(v, file, pickle.HIGHEST_PROTOCOL)