import os
import json
import torch

__all__ = ['get_gpu_prop', 'collect_gpu_info', 'show_gpu_info']


def get_gpu_prop(show=False):
    ngpus = torch.cuda.device_count()
    
    properties = []
    for dev in range(ngpus):
        prop = torch.cuda.get_device_properties(dev)
        properties.append({
            'name': prop.name,
            'capability': [prop.major, prop.minor],
            'total_momory': round(prop.total_memory / 1073741824, 2), # unit GB
            'sm_count': prop.multi_processor_count
        })
       
    if show:
        print('cuda: {}'.format(torch.cuda.is_available()))
        print('available GPU(s): {}'.format(ngpus))
        for i, p in enumerate(properties):
            print('{}: {}'.format(i, p))
    return properties


model_name = 'maskrcnn_resnet50_pan'
dirname = os.path.dirname(__file__)
json_file = os.path.join(dirname, 'gpu_info.json')


def sort(d, tmp={}):
    for k in sorted(d.keys()):
        if isinstance(d[k], dict):
            tmp[k] = {}
            sort(d[k], tmp[k])
        else:
            tmp[k] = d[k]
    return tmp


def collect_gpu_info(fps):
    fps = [round(i, 2) for i in fps]
    if os.path.exists(json_file):
        gpu_info = json.load(open(json_file))
    else:
        gpu_info = {}
    
    prop = get_gpu_prop()
    name = prop[0]['name']
    check = [p['name'] == name for p in prop]
    if all(check):
        count = str(len(prop))
        if name in gpu_info:
            gpu_info[name]['properties'] = prop[0]
            perf = gpu_info[name]['performance']
            if count in perf:
                if model_name in perf[count]:
                    perf[count][model_name].append(fps)
                else:
                    perf[count][model_name] = [fps]
            else:
                perf[count] = {model_name: [fps]}
        else:
            gpu_info[name] = {'properties': prop[0], 'performance': {count: {model_name: [fps]}}}

        gpu_info = sort(gpu_info)
        json.dump(gpu_info, open(json_file, 'w'))
    return gpu_info
    

def show_gpu_info(dataset='coco 2017', reduction='max'):
    gpu_info = json.load(open(json_file))

    datasets = {'coco 2017': (117266, 4952), 'voc 2012': (1463, 1444)}
    prices = {
        '1x GeForce GTX 1070': (2.8,),
        '1x GeForce GTX 1070 Ti': (2.58,),

        '1x GeForce GTX 1080': (3,),
        '1x GeForce GTX 1080 Ti': (4.8, 5, 5.6),
        '2x GeForce GTX 1080 Ti': (6.5, 6.8),
        '4x GeForce GTX 1080 Ti': (13,),

        '1x GeForce RTX 2080': (4,),
        '1x GeForce RTX 2080 Ti': (6,),
        '2x GeForce RTX 2080 Ti': (6.6, 7.5),
        '4x GeForce RTX 2080 Ti': (16.1,),

        '1x TITAN RTX': (6.8, 7.1),
        '2x TITAN X': (5,),
        '2x TITAN Xp': (6,),
    }

    print('         GPU              hour/epoch    yuan/epoch    index')
    d = datasets[dataset]
    for k, v in gpu_info.items():
        for c, p in v['performance'].items():
            instance = '{}x {}'.format(c, k)
            if instance not in prices:
                print(instance)
            else:
                if reduction == 'max':
                    perf = [max(fps) for fps in zip(*p[model_name])]
                elif reduction == 'min':
                    perf = [min(fps) for fps in zip(*p[model_name])]
                    
                time_per_epoch = (d[0] / perf[0] + d[1] / perf[1]) / 3600
                
                price = sum(prices[instance]) / len(prices[instance])
                cost = price * time_per_epoch
                index = cost * time_per_epoch
                print('{:23s}:    {:5.2f}         {:5.2f}        {:.0f}'.format(instance, time_per_epoch, cost, index))