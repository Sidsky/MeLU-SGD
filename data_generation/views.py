import json
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from data_generation.generate_data import getData
from recommender_system.views import train


# Create your views here.
def home(request):
    return render(request, 'home.html')


def about(request):
    return render(request, "about.html")


@csrf_exempt
def generateData(request):
    allData = getData()
    movie_records = allData.items.reset_index().to_json(orient='records')
    user_records = allData.users.reset_index().to_json(orient='records')
    score_records = allData.score.reset_index().to_json(orient='records')

    movie_data = json.loads(movie_records)
    user_data = json.loads(user_records)
    score_data = json.loads(score_records)

    context = {'m': movie_data, 'u': user_data, 's': score_data}

    return render(request, "index.html", context)


def train_from_generate(request):
    return train(request)
