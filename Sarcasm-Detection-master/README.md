SARCASM DETECTION

Files:

nlp_report.pdf : Report of the project with all the details related to development.

GetTweets.py - Running this will generate the sarcastic or non sarcastic tweet data based on the query
nonsarcasmfull.csv - This contains all the non sarcastic tweet data generated by GetTweets.py
sarcasmfull.csv - This contains all the sarcastic tweet data generated by GetTweets.py

preprocess.py - takes the above csv files and preprocesses it and generates clean data
nonsarcpreproc.npy - clean non sarcastic tweet data generated by preprocess.py
sarcpreproc.npy - clean sarcastic tweet data generated by preprocess.py

feature_extraction.py - Takes an input string and genearates all the feature required for training and testing
training.py - takes the clean tweet data files and creates feature set for each tweet and trains it 
	      and stores the classifier in the file.
classif_all.p - classifier which is trained by the training.py is stored here
vecdict.p - This is the dictionary vector object to convert from the lists of feature value to vectors to train or test

sarcasm.py - This is the main program to check sarcasm score. It uses classif_all.p and vecdict.p to to extract featureset
       from the new input sentence and clasifies to either sarcastic or non sarcastic and gives a quantitative score


Running the program:

$ python sarcasm.py
// And enter the input to console to get output score

example:
$ python sarcasm.py 
enter the sentence
Donald trump will make america great again
18
enter the sentence
Messi is the best footballer in the world
-70

