# Data and Replication Code for Human Detection of Machine-Manipulated Media

## Data

The `tabular-data` folder contains 3 .csv files. Participant responses can be found in the deepangel.csv. Image level characteristics can be found in image_characteristics.csv. Subjective quality ratings of the target object removal AI can be found in subjective_quality_rankings.csv.

The `image-pairs` folder contains images included in the two alternative forced choice experiment. These images were all uploaded by website visitors, and these visitors indicated that they would like to share these images publicly. 

## Replication Script

```
pip install -r requirements.txt
mkdir plots
mkdir regression
```

Then, run

```
python replication_analysis.py fe all all
```

