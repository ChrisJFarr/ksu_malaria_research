import pandas as pd

"""
Additional data retrieved from http://www.cheminfo.org/flavor/malaria/index.html
From OSM and contains all compounds in their database
SDF File parser, might have been used to create the file
https://github.com/Actelion/openchemlib/tree/4633da5f2bfc3fbd59e2e01897c42d7be2b22b2d/src/main/java/com/actelion/research/chem/io


TODO work to understand the target IC50
https://www.graphpad.com/support/faqid/1356/
Studies are done In vitro since using IC50 (EC50 is In vivo)
https://en.wikipedia.org/wiki/In_vitro
Caution: TODO research this further to learn more about what would be different?
https://en.wikipedia.org/wiki/IC50 (Functional antagonist assay)
IC50 values are very dependent on conditions under which they are measured. 
Question: For our dataset, are all IC50 data-points collected under similar conditions or is this something
that we need to adjust for?

Question: Are we considering toxicity of compounds in our search? What data points should be leveraged for this?


Existing Machine Learning Approaches
https://malariajournal.biomedcentral.com/articles/10.1186/1475-2875-10-274


"""


df = pd.read_csv("data/Series3_6.15.17_padel.csv")
df.head()

# Which have missing values? How do they show when clustering?
df[df.IC50.isnull()]


df = pd.read_csv("data/Selleck_filtered_padel_corrected.csv", encoding='cp1252')
akt1 = pd.read_csv("data\Akt1_decoys_padel.csv")

akt1.head()

# Read sdf file to view contents
import sdf
sdf_file = sdf.load("data/malaria-2018-04-16.sdf", '/v')
# SDF File parser https://github.com/Actelion/openchemlib/tree/4633da5f2bfc3fbd59e2e01897c42d7be2b22b2d

df = pd.read_csv("data/result-2018-04-16.txt", delimiter="\t")
df.head()


len(df)




