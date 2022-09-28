# TensorFlowTweetGenerator
Generates text based on previous tweets of a politician.

Dataset came from: https://www.kaggle.com/shivammehta007/trump-tweetcsv

- main.py 
   - loads the trained tensorflow model, so that text generations can be made without having to re-train the model each run
   - When you run main.py, the program will load the model and prompt the user in the console to type a starter sentence:
      - ex. Enter the starting text: the election was a
            the election was a total hoax and the dems not a party apparatus ” thank about on information for his more an long before
      - The model isn't perfect, but it can generate some interesting comments.
- ModelCreation.py
   - Code for pre-processing the data and creating the tensorflow model. 

I deployed the model using heroku. If you want to test it out, check out https://politician-tweet-generator.herokuapp.com/
