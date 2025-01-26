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

**Due to limited computational resources, the dataset is divided into smaller subsets, with 50% allocated for `training`, `dev_test` and `test`.**

### 2. Simple Siamese NN 
<details>
  <summary>Click to view: Build and Train the Siamese Model :</summary>
  
```python
# Function to build and train the Siamese model
def train_siamese_model(input_shape, dense_layer_sizes, anchor_input, positive_input, negative_input, X_train, X_val):

    # Create the shared network
    shared_network = create_siamese_model(input_shape, dense_layer_sizes, dropout_rate=0)

    # Create the embeddings
    anchor_embedding = shared_network(anchor_input)
    positive_embedding = shared_network(positive_input)
    negative_embedding = shared_network(negative_input)

    # Compute the distances
    distances = DistanceLayer()([anchor_embedding, positive_embedding, negative_embedding])

    # Create the Siamese model
    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)

    # Compile the model
    siamese_model.compile(optimizer='adam', loss=triplet_loss)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    siamese_model.fit([X_train['anchor'], X_train['positive'], X_train['negative']], np.ones(len(X_train['anchor'])), 
                       validation_data=([X_val['anchor'], X_val['positive'], X_val['negative']], np.ones(len(X_val['anchor']))),
                       epochs=3, 
                       batch_size=32, 
                       callbacks=[early_stopping])
    return siamese_model
```
<\details>
