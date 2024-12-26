from summa import keywords
from summa.summarizer import summarize

# Define the texts
text1 = """
Neo-Nazism consists of post-World War II militant social or political movements seeking to revive and implement the ideology of Nazism. 
Neo-Nazis seek to employ their ideology to promote hatred and attack minorities, or in some cases to create a fascist political state. 
It is a global phenomenon, with organized representation in many countries and international networks. 
It borrows elements from Nazi doctrine, including ultranationalism, racism, xenophobia, ableism, homophobia, anti-Romanyism, antisemitism, anti-communism, and initiating the Fourth Reich. 
Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. 
In some European and Latin American countries, laws prohibit the expression of pro-Nazi, racist, anti-Semitic, or homophobic views. 
"""

text2 = """
Prototypes are widely recognized to be a core means of exploring and expressing designs for interactive computer artifacts. 
It is common practice to build prototypes in order to represent different states of an evolving design, and to explore options. 
However, since interactive systems are complex, it may be difficult or impossible to create prototypes of a whole design in the formative stages of a project. 
Choosing the right kind of more focused prototype to build is an art in itself, and communicating its limited purposes to its various audiences is a critical aspect of its use.
"""

text3 = """
There are around fifty survey articles published in recent years that deal with 4G and 5G cellular networks. 
From these survey articles only seven of them deal with security and privacy issues for 4G and 5G cellular networks. 
The article highlights several similarities and differences between 4G and 5G systems, such as the use of mutual authentication and increased configurability in 5G for security features. 
For example, 5G ensures the permanent subscription identifier is never sent in clear text over the air, unlike 4G. 
"""

# Define a function for summarizing and extracting keywords
def process_text(text, ratio, top_keywords):
    print("Summary:")
    print(summarize(text, ratio=ratio))
    print("\nKeywords:")
    print(keywords.keywords(text, words=top_keywords))
    print("\n" + "-"*50 + "\n")

# Process each text
print("Text 1:")
process_text(text1, ratio=0.3, top_keywords=5)

print("Text 2:")
process_text(text2, ratio=0.3, top_keywords=5)

print("Text 3:")
process_text(text3, ratio=0.3, top_keywords=5)
