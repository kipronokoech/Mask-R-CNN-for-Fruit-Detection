import json
import json
import re
import os
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from ast import literal_eval

event_files = [i for i in os.listdir("./event_files/") if "event_files" in i]

for folder in event_files:
	if os.path.exists("history/{}.json".format(folder.split("_")[-1])) and os.path.exists("history/{}.csv".format(folder.split("_")[-1])):
		continue
	all_details = []
	print("Processing the following folder:", folder)
	for file in os.listdir(os.path.join("./event_files/", folder)):
		for summary in summary_iterator(os.path.join("./event_files/", folder, file)):
			for v in summary.summary.value:
				data = {
				"epoch": summary.step,
				"tag" : v.tag,
				"step_value": v.simple_value
				}
				all_details.append(data)

	all_details = sorted(all_details, key= lambda i:i["epoch"])
	
	f = open("history/{}.json".format(folder.split("_")[-1]),"w+")
	json.dump(all_details,f,indent=3)
	f.close()
	df = pd.DataFrame(all_details)
	df.to_csv("history/{}.csv".format(folder.split("_")[-1]))