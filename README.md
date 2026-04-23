Step 1 — Install extra libraries (run this once at the top of your notebook)

!pip install imbalanced-learn statsmodels scikit-learn 
seaborn

Step 2 — Upload your datasets by running this cell:

from google.colab import files
uploaded = files.upload()

Then select all your CSV files at once.
