# LSH with Extra Trees
This code is created for the "Scalable Product Duplicate Detection" assignment of the course Computer Science for Business Analytics at the Erasmus University Rotterdam. The goal is to find a scalable solution for duplicate product detection using data coming from several Web shops. To do this, we need to reduce the number of comparisons using the approximation technique of Locality Sensitive Hashing (LSH).
## Code structure
The code is structured in the following way. First, we load the .json file and write a function that loops over the nested dictionary. Next, we clean the data by removing unnecessary values and words. For every word in the titles, we count the number of occurences and remove the words that occur once. With the final titles, a matrix is created, consisting of binary vectors representing the product titles. Titles that have no words are removed. Next, we define the similarity measure and use this to for the LSH algorithm. After this a new matrix is made as input for the Extra Trees algorithm. Lastly, the Extra Trees algorithm with five bootstraps is used as duplicate detection method.
## Usage
After importing the data file and the necessary libraries (pandas, numpy, etc.) via [pip](https://pip.pypa.io/en/stable/), the code can be run without any further complications.
```python
# Load data
f = open("dataset.json")
data = json.load(f)
```
