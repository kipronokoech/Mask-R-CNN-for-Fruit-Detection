- There is an hidden file names .junk that contains the code that was used to build the code in this folder but are not useful after some time. After some time this foder will be deleted. Dated Oct 9, 2020.

- Evaluation.py contains a class that defines all the metrics used in the project: Confusion matrix, AP, Precision and Recall.

- MaskReconstruction.py - This script contains all functions related to manipulation of model output from contour reconstruction, drawing and writing contors.

-runMain.py - Running this matrix call the class in Evaluation.py, that is, the script is mainly used to generate and save the results.

- generate_truth_masks.py - This script is used to generate the annotations/labels for each image. This is important for the purposes of the evaluation.

- Plot_PR_and_test_masks.py - In a high level this script is not esential for the purpose of reproducibility fof the results. This files is used to generate examples and to plot precision-recall curve for the purpose of the usage in the research paper.

masks - this folder contains the masks - truth and predictions - for both training and testing set.

results - contains all results of the model perfomance based on the metrics used- values and plots.

Any other file in this folder can be regarded as inconsequential and was probably used just for testing concepts and ideas.




