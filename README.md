# __<center>Biomedical Text Analysis Using Deep Neural Networks</center>__

## __<center>Overview</center>__
[This project](https://github.com/VivianNg9/Biomedical-Text-Analysis-Using-Deep-Neural-Networks/blob/main/Biomedical%20Text%20Analysis%20Using%20Deep%20Neural%20Networks.ipynb) focuses on query-focused summarisation for medical questions. The goal is to determine whether a sentence extracted from relevant medical publications can be used as part of the answer to a given medical question.

## __<center>Dataset</center>__
[`data.zip`](https://github.com/VivianNg9/Biomedical-Text-Analysis-Using-Deep-Neural-Networks/blob/main/data.zip)
Using data that has been derived from the **BioASQ challenge** (http://www.bioasq.org/), after some data manipulation to make it easier to process for this assignment. 
The BioASQ challenge organises several "shared tasks", including a task on biomedical semantic question answering which we are using here. 
Utilising a labeled dataset (`bioasq10_labelled.csv`) with:
- **Questions:** Medical queries.
- **Sentences:** Extracted text from relevant publications.
- **Labels:** Binary labels indicating relevance (1 for relevant, 0 for not relevant).

## __<center>Project Workflow</center>__
### 1. Data Review 
![Data Review](https://github.com/VivianNg9/Biomedical-Text-Analysis-Using-Deep-Neural-Networks/blob/main/image/DataReview%20.png)

The columns of the CSV file are:
* `qid`: an ID for a question. Several rows may have the same question ID, as we can see above.
* `sentid`: an ID for a sentence.
* `question`: The text of the question. In the above example, the first rows all have the same question: "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
* `sentence text`: The text of the sentence.
* `label`: 1 if the sentence is a part of the answer, 0 if the sentence is not part of the answer.
