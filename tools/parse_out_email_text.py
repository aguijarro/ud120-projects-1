#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
#import nltk
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        ###words = text_string

        ### split the text string into individual words

        #tokenized_text = nltk.word_tokenize(text_string)

        ### stem each word

        stemmer = SnowballStemmer("english")

        #for word in tokenized_text:
        stems = []
        for word in text_string.split():
            stems.append(stemmer.stem(word))

        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        words = " ".join(stems)
    return words

