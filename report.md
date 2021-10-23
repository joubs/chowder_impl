# Description of the implemented algorithm.

The CHOWDER algorithm falls under the scope of weak supervised learning. Its primary objective is to predict a binary 
label (Healthy VS Cancer) out of a given image. 
It operates over WSI images, that have a very high resolution. While the data is annotated at the image level, the input 
data is sampled in tiles. The algorithm exploits tiles descriptors to predict a label at the image level. 

The algorithm can also be used to predict a label at the tile level, a usage could be for example to help a clinician to
annotate a dataset by suggesting suspicious regions on a WSI image.
  

# Design choices and specifications of the code.

## Code specs 

The proposed application shall:
  - Obtain a path to medical data from the user, and load it in memory from disk.
  - train a classifier with CHOWDER method with training data and labels. The metric used to assess the performance of
  the classifier should be the AUC. 
  - output evaluation results over testing data as a csv file called "test_output.csv"

The proposed application should: 
  - log in console, and store on disk training and evaluation insights such as intermediate performance. 
  - perform early stopping to keep the best trained model and store it on disk. 


## Design choices 
  The following design choices have been made, essentially to save time: 
  - The slide data is zero padded, to handle the variation of the number of tiles, so that the batches have identical
  shapes 
  - The CHOWDER method has been implemented without dropout and l2 regularisation. 
  - The config parameters are written directly in the main script, they should ideally be written in a proper config
  file (in yaml for example)

# Experimental results.


| Model              | AUC score |
| :-----------------:|----------:|
| Baseline+maxpool   |    0.65   |
| CHOWDER (R=5)      |    0.857  |
| CHOWDER (R=10)     |    0.837  |


# Suggestions of improvement.

- Add a proper config file 
- Try the 3 fold cross-validation 
- Test on another dataset to assess the robustness
- More hyperparameters optimisation(grid search or random search)
- Add more unit tests 