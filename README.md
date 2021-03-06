# Model fidelity map (MFmap)

The MFmap is a semi-supervised generative model integrating omics data, matching cell lines to cancer subtypes. 
Publication based on MFmap:
- [MFmap: A semi-supervised generative model matching cell lines to tumours and cancer subtypes] (https://www.biorxiv.org/content/10.1101/2021.07.15.452446v1.full)

## MFmap web app 
MFmap shiny app is publically available at (http://h2926513.stratoserver.net:3838/MFmap_shiny/)

## MFmap methodology overview
<div>
    <img src="media/overview.png" width=1000>
</div>

### Prepare input data for MFmap.
The original mutation and CNV data are represented as a binary matrix indicating the presence/absence of a DNA alteration in a cell line or a bulk tumour sample. This sparse binary matrix is projected onto a cancer reference network ([Huang et al.,Bioinformatics 2018](http://academic.oup.com/bioinformatics/article/34/16/2859/4956012)) and network diffusion algorithm is used to propagate the signals of single somatic events onto their network neighbours, resulting in continuous matrix with the same dimensions of the original binary matrix.
### The architecture of MFmap. 
MFmap has three components: an encoder, a classifier and a decoder, encoded by different colours in the above figure. Input layer of the encoder contains two views, network smoothed mutation and CNV data are concatenated and input into DNA view, RNA view inputs gene expression data. The encoder maps the input data of each sample into a latent representation vector <img src="https://render.githubusercontent.com/render/math?math=\vec{z}". The decoder uses  <img src="https://render.githubusercontent.com/render/math?math=\vec{z}" to reconstruct the DNA and RNA views in its output layer. Subtypes of bulk tumours are used to train the classifier. During training, the classifier predicts subtypes of cell lines. 
### Visualising results output by MFmap.
To organise and summarise sample associations we used the visualisation concept of OncoGPS([Kim et al.,Cell Syst. 2017](https://www.cell.com/cell-systems/fulltext/S2405-4712(17)30335-6)). The biological meanings of latent representations are annotated and MFmap reference map is generated. Both steps are based on bulk tumour latent representations learnt by MFmap. Samples can then be projected onto the MFmap reference map to visualise sample properties such as drug sensitivity and subtypes, complemented with the information of sample relationship.

## Use MFmap
Clone this repository to use MFmap
```bash
git clone https://github.com/mcmzxx/mfMap.py.git
cd mfMap.py

# run the example using simulated data
bash run-example.sh

# run the example using realistic colon cancer data
mkdir data_bak
cd data_bak
# download the data from data repository (https://cloud.hs-koblenz.de/s/WFWjMq9pJ8i29WD)
cd ..
bash run_example_real_data.sh
```

### Input Data
MFmap takes inputs of datasets containing both cell lines bulk tumours: (1) gene expression profile which is a gene by sample matrix, (2) DNA profile which is a gene by sample matrix, and (3) cancer subtype labels of bulk tumours.

In __gene expression profile__ or __DNA profile__, values should be separated by tabs.

| barcode | TCGA.A6.2678.01 | TCGA.AA.3950.01 | TCGA.DM.A1HB.01 | CL40_LARGE_INTESTINE | SW403_LARGE_INTESTINE | SNUC4_LARGE_INTESTINE |
|  :-------------: |  :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| HSPA2 | 0.178085 | 0.573315 | 0.503795 | 0.547310 | 0.243164 | 0.495841 |
| HSPA6 | 0.337385 | 0.556164 | 0.487331 | 0.531813 | 0.550296 | 0.784094 |
| HIST1H4I | 0.108904 | 0.511286 | 0.450405 | 0.488413 | 0.400680 | 0.446989 |
| HSPA8 | 0.331599 | 0.600488 | 0.503966 | 0.562086 | 0.411304 | 0.495323 |
| B2M | 0.399159 | 0.561251 | 0.518686 | 0.608198 | 0.240596 | 0.479772 |

__Cancer subtype label__ list has the following three columns, separated by tabs.
| barcode | subtype | type |
|  :-------------: |  :-------------: | :-------------: |
| TCGA.A6.2678.01 | NOLBL | tumor |
| TCGA.AA.3950.01 | CMS1 | tumor |
| TCGA.DM.A1HB.01 | CMS1 | tumor |
| SNUC4_LARGE_INTESTINE | NOLBL | cell |
| SW403_LARGE_INTESTINE | NOLBL | cell |
| CL40_LARGE_INTESTINE | NOLBL | cell |
