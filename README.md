# cord19m
Solution for COVID-19 Open Research Dataset ([CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)) 

This utilises documents selection work done in [biobert-metadata-serch](https://www.kaggle.com/sourojit/cord-biobert-search/) and [reference-title-cluster](https://www.kaggle.com/debasmitadas/unsupervised-clustering-covid-research-papers) notebooks on kaggle

### Installation
You can use Git to clone the repository from GitHub and install it. It is recommended to do this in a Python Virtual Environment. 

    git clone https://github.com/Aakash5/cord19m.git
    cd cord19q
    pip install requirements.txt

Copy config.example.ini file to config.ini , update path variables

### Building a summary file

	Run Summary-by-task.ipynb

### Running a QA file

	Run Try.ipynb
