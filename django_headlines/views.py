from django.shortcuts import render        
from django.http import HttpResponse
import json
import subprocess
from experiments.x08_morphing.test import getinput,findscore
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
@csrf_exempt 
def query(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_text = data['input']
        print(input_text)
        # process = subprocess.Popen(['python3', './experiments/x08_morphing/test.py', input_text, task], stdout=subprocess.PIPE)
        op = findscore(input_text)
        # output, error = process.communicate()
        print(f"score:{op}")
        return HttpResponse(json.dumps(op),content_type='application/json')
    else:
        return HttpResponse(status=405)
    
@csrf_exempt 
def getquery(request):
    if request.method == 'POST':
        print('hiiii')
        data = json.loads(request.body)
        task = data['task']
        # process = subprocess.Popen(['python3', './experiments/x08_morphing/test.py'], stdout=subprocess.PIPE)
        # output, error = process.communicate()
        output =  getinput()
        print('input sentence is')
        print(output)
        response = {'output':output[0],'id':output[1]}
        return JsonResponse(response)
    else:
        return HttpResponse(status=405)
    
def index(request):
    return HttpResponse("Hello, world!")


@csrf_exempt 
def postresult(request):
    if request.method == 'POST':
        result_dict = json.loads(request.body.decode('utf-8'))  # Assuming the data is sent as form data
        print(result_dict)
        del result_dict['time']

        with open('../results.txt', 'a') as file:
            file.write(json.dumps(result_dict))
            file.write('\n') 
        response_data = {'message': 'Data received successfully'}
        return HttpResponse(response_data)