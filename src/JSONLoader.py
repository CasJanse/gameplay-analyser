import json
import os


def load_json_file(dir):
    json_content_dict = {}
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            with open(dir + filename) as json_file:
                accuracy = round(float(json.load(json_file).split("\'accuracy\': [")[1][:-2].split(" ")[-1]) * 100, 2)
                json_content_dict.update({filename[:-len("_history.json")]: accuracy})
    return json_content_dict


data = load_json_file("../../networks/multi-video-greyscale-20/")
print(data)
