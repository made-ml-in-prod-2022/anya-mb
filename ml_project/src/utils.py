from os.path import join
import pickle


def save_object(obj, path, filename):
    filepath = join(path, filename)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_object(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj