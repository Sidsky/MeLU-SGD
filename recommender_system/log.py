import matplotlib.pyplot as plt
import pickle
import base64
import numpy as np
from io import BytesIO


def get_plots():
    training_loss = pickle.load(open("recommender_system/training_loss.pkl", "rb"))
    validation_loss = pickle.load(open("recommender_system/validation_loss.pkl", "rb"))

    epochs = [e+1 for e in range(25)]

    plt.switch_backend('AGG')
    plt.figure(figsize=(5, 4))
    plt.plot(training_loss, epochs, 'm')
    plt.xlabel("Number of epochs")
    plt.ylabel("Training Loss")
    plt.title("Training loss vs. Number of epochs")
    plt.tight_layout()
    training_graph = get_graph()

    plt.switch_backend('AGG')
    plt.figure(figsize=(5, 4))
    plt.plot(validation_loss, epochs, 'g')
    plt.xlabel("Number of epochs")
    plt.ylabel("Validation Loss")
    plt.title("Validation loss vs. Number of epochs")
    plt.tight_layout()
    validation_graph = get_graph()
    training_loss = training_loss[-1]
    validation_loss = validation_loss[-1]
    return training_graph, validation_graph, training_loss, validation_loss


def get_graph():
    buffer = BytesIO()

    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()

    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')

    buffer.close()
    return graph


class log:
    def __init__(self):
        self.training_loss = []
        self.validation_loss = []
        self.test_loss = []

    def get_training_loss_curve(self, epochs):
        p_training_loss = pickle.load(open("training_loss.pkl", "rb"))
        p_validation_loss = pickle.load(open("validation_loss.pkl", "rb"))

        plt.plot(p_training_loss, 'm')
        plt.xlabel("Number of epochs")
        plt.ylabel("Training Loss")
        plt.title("Training loss vs. Number of epochs")
        plt.show()

        plt.plot(p_validation_loss, 'g')
        plt.xlabel("Number of epochs")
        plt.ylabel("Validation Loss")
        plt.title("Validation loss vs. Number of epochs")
        plt.show()

        plt.plot(p_training_loss, 'm', label="Training Loss")
        plt.plot(p_validation_loss, 'g', label="Validation Loss")
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs. Number of epochs")
        plt.legend()
        plt.show()

    def store_on_disk(self):
        pickle.dump(self.training_loss, open("recommender_system/training_loss.pkl", "wb"))
        pickle.dump(self.validation_loss, open("recommender_system/validation_loss.pkl", "wb"))

    def get_length(self):
        print(len(self.training_loss))
        print(self.training_loss)
        print()
        print(len(self.validation_loss))
        print(self.validation_loss)


if __name__ == '__main__':
    logger = log()
    get_plots()
