import json
import os
import pandas as pd


set1 = "fruits2498"

confusions = [i for i in os.listdir("results/{}/".format(set1)) if "ConfusionMatrix" in i]
# a = json.load(open("./results/trainConfusionMatrix%0.85%0.3.json"))
# quit()
if os.path.exists("summary_results_{}.json".format(set1)):
	print("summary_results_{}.json".format(set1), "already exist.Qitting.")
	quit()
	# os.remove("summary_results_{}.json".format(set1))

summary_stats = []
for confusion in confusions:
	print(confusion)
	sett = confusion.split("ConfusionMatrix")[0]
	_, confidence, iou_string = confusion.split("%")
	iou = float(iou_string.replace(".json",""))
	confidence = float(confidence)
	try:
		summary = json.load(open("./results/{}/{}".format(set1,confusion)))[0]
		summary["Confidence"] = confidence
		summary["IOU"] = iou
		summary["Set"] = sett
		summary_stats.append(summary)
	except:
		continue

f = open("summary_results_{}.json".format(set1),"a+")
json.dump(summary_stats,f,indent=3)
f.close()

df = pd.DataFrame(json.load(open("summary_results_{}.json".format(set1))))
df.sort_values(by=["Confidence","IOU","Set"],inplace=True, ascending=False)
df = df[['Confidence', 'Set', 'IOU','TP(%)','FP(%)','FN',
        'Total TP', 'Total FP', 'TOtal FN','All Detections', 'All GT']]
df.to_csv("summary_results_{}.csv".format(set1), index=False)

