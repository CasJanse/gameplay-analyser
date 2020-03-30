import JSONLoader
import matplotlib.pyplot as plt


json_dict = JSONLoader.load_json_file("../../networks/accuracy-test/")

plt.bar(range(len(json_dict)), list(json_dict.values()), align="center")
plt.xticks(range(len(json_dict)), list(json_dict.keys()))
plt.show()
