# Final Project - Class Competition
CS 434 - Machine Learning & Data Mining
Tarren Engberg - _sole group member_
engbergt@oregonstate.edu
June 12th, 2019

## Tasks
The goal of the competition is to build the most accurate model for detecting the presence/absence of Pseudoknot. Each task may have up to three submisssions.
1. Designed model can only use the selected 103 features to make predictions.
2. Designed model can consider all 1053 features.

## Discovery

### Data

#### Raw fasta RNA sequences 
_id line, sequence line_
* pks_Train.fasta.txt (present)
* pkfs_Train.fasta.txt (free)

#### Feature data
_Tab separated
header line first
Id, pk (0 free, 1 present)
pks=pk free, pk=pk present_
* featuresall_train.txt
* feature103_Train.txt (subset)

### Assumptions

#### data
* bpRNA_CRW_10025 is the first to have knots
* bpRNA_CRW_10044 is the first to be free of knots
* featuresall_train.txt file lists knot present examples first, then free examples.

## Design



## Report

### 1 - Data preprocessing
_Did you pre-process your data in any way? This can be for the purpose of reducing dimension, or reducing noise, or balancing the class distribution. Be clear about what you exactly did. The criterion is to allow others to precisely replicate your works._

### 2 - Learning Algorithms

#### 2.1 - Methods explored
_Provide a list of learning algorithms that you explored for this project. For each algorithm, briefly justify your rationale for choosing this algorithm._


#### 2.2 - Final models
_What are the final models that produced your submitted test predictions? Be sure to provide enough detail so that by reading the description, the model can be recreated._

### Parameter Tuning & Model Selection

#### 3.1 - Tuning
_What parameters did you tune for your models? How do you perform the parameter tuning?_

#### 3.2 - Selection
_How did you decide which models to use to produce the final predictions? Do you use cross-validation or hold-out for model selection? When you split the data for validation, is it fully random or special consideration went into forming the folds? What criterion is used to select the models?_


### 4 - Results
_Do you have any internal evaluation results you want to report?_

### 5 - Resources