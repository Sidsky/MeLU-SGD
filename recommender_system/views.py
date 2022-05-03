from django.shortcuts import render
from recommender_system.train import generate_and_train
from recommender_system.test import test
from recommender_system.log import get_plots
from recommender_system.config import states


# Create your views here.

def train(request):
    MovieLens_1M = 'data_generation/movie_lens_dataset'
    generate_and_train(MovieLens_1M)
    training_curve, validation_curve, training_loss, validation_loss = get_plots()

    context = dict(training=training_curve, validation=validation_curve, train_loss=training_loss,
                   val_loss=validation_loss)

    for state in states:
        context[state] = test(MovieLens_1M, state)

    return render(request, 'train.html', context)
