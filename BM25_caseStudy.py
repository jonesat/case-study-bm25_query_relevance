### Modules
import os,errno,re,nltk,string,glob,json,math
from stemming.porter2 import stem

from pprint import pprint

# The following code parses documents to prepare them for ranking their relevance in queries.
# It does this by creating various python classes, to parse, contain, rank and write document information.
# All of this work depends on some understanding of what is being parsed.
# We are parsing a word, which I will take to be defined as a unit element of a 
# language that has some meaning.
# Compared to a term in a document which I will take as the simplest form of a 
# non-common word that has a minimum number of characters, 
# that additionally still carries some meaning

### Node and Linked List Classes
# This is the primary data structure we will be using to manage our documents.
class Node:    
    '''
    A node in a linked list.
    Attributes: data, (any): This is what is stored at the node
    Attributes: next, (Node): Optionally specifies the next node in the linked list
    '''
    def __init__(self,data:any,next=None) -> None:
        '''
        Initialises a new node.
        Inputs: data, any: The data to store in the node.
        Inputs: next, The next node in the linked list.
        Returns: None
        '''
        self.data=data
        self.next=next

class LinkedList:
    '''
    A linked list that contains nodes.
    Attributes: hnode: The head node, the first element of the linked list.
    '''
    def __init__(self,hnode:Node)->None:
        '''
        Initialises the linked list
        Inputs: hnode, the first node of the linked list.
        Returns: None.
        '''
        self.head=hnode
    
    def insert(self,nnode:Node) -> None:
        ''' 
        This method adds a new node to the linked list
        Inputs: nnode, a new node.       
        Returns: None
        '''
        
        if self.head!=None:
            p=self.head
            while p.next !=None:
                p=p.next
            p.next=nnode
    
    def traverse(self,function1:callable,function2:callable)->any:
        '''
        An experimental method created before I discovered the __iter__() dunder method.
        It traverse the linked list and performs a function call on each node, and accumulates the result in another function
        Inputs:  function1, callable that accumulates the results of function calls on each node
        Inputs: function2, callable that is applied to each node
        Returns: any, whatever the result of the accumulation of each visted node is.
        '''
        accumulator = function1()
        node = self.head
        while node != None:
            accumulator = function2(node,accumulator)
            node = node.next
        return accumulator

    def lprint(self):
        '''
        This function prints the data of each node in a linked list, separating them with arrows to indicate the direction
        Inputs: None
        Returns: None
        '''
        outputString="";
        if self.head != None:
            p = self.head
            while p!=None:
                outputString+=f"{p.data} "
                if p.next != None:
                    outputString+="--> "
                p = p.next
            print(outputString)
        else:
            print("The list is empty")
    
    def __len__(self)->int:
        '''
        This dunder method allows a linked list to have a length property.
        It is calculated using the experimental traverse method to count node visists.
        Inputs: None
        Return: length, integer.
        '''
        f1= lambda : 0
        f2= lambda node,accumulator: accumulator+1
        return self.traverse(f1,f2)

    def __iter__(self)->Node:
        '''
        Use as a replacement for the experimental traverse method it allows
        A linked list to function as a standard iterable.
        '''
        node = self.head
        while node is not None:
            yield node
            node = node.next

### DocWords Class
class DocWords():
    '''
    This class captures the contents of an xml file in the bag of words format.
    Attributes: docID, str, a value that identifies the document
    Attributes: title, string, the title of a document
    Attributes: terms, dictionary, a dictionary of tokenized words and their corresponding frequencies in the document
    Attributes: wordCount, int, the total number of words, not terms in the dictionary

    '''
    def __init__(self,docID:str,terms:dict,wordCount:int,title:str)->None:
        '''
        The function initializes a new DocWords object
        Inputs: docID, str, a value that identifies the document
        Inputs: title, string, the title of a document
        Inputs: terms, dictionary, a dictionary of tokenized words and their corresponding frequencies in the document
        Inputs: wordCount, int, the total number of words, not terms in the dictionary        
        Returns: None
        '''
        self.docID = docID
        self.title = title
        self.terms = terms
        self.wordCount = wordCount

    def addTerm(self,term:str,stopWords:list)->None:
        '''
        This function builds a dictionary of terms and term frequencies after stemming an input term.
        Inputs: term, str, a word from a text document that needs to be stemmed before being considered a term
        Inputs: stopWords, list, a list of common english words that don't contribute to meaning.
        Returns: None
        '''
        MINWORDLENGTH = 2
        term = stem(term.lower())        
        if len(term)>MINWORDLENGTH and term not in stopWords:
            try:
                self.terms[term]+=1
            except KeyError:
                self.terms[term]=1                

    def DictSortbyKeys(self)->None:
        '''
        This is a function that sorts the terms dictionary of a docWords object alphabetically
        '''
        return {k:v for k,v in sorted(self.terms.items(),reverse=False)}
   
    def DictSortbyFreq(self)->None:
        '''
        This is a function that sorts the terms dictionary of a docWords object by frequency in a descending order
        '''
        return {k: v for k, v in sorted(self.terms.items(),key=lambda term: term[1],reverse=True)}

    def GetTuple(self):
        '''
        This function returns a tuple that displays the document ID along with a term-frequency dictionary of a docWords object
        '''
        return(self.wordCount,{self.docID:self.DictSortbyFreq()})
    
    def Plus(self):
        '''
        This function increments the word count of the docWords object as it is called.
        This function services the requirements of Task 3.1 by tracking the "length" of a document.
        '''
        self.wordCount+=1
    
    def __repr__(self)->str:
        '''
        This dunder method determins how a docWords object interacts with str() and print() methods
        Returns: str, A string representation of a docWords object
        '''
        return f"Document itemid: {self.docID}, contains: {self.wordCount} words and {len(self.terms.keys())} terms"

    def __len__(self)->int:
        '''
        This method determines how a docWords object interacts with the len() method.
        Returns the number of terms in a docWords object, compared to the number of words.
        '''
        return len(self.terms)

### Document Parser Class
class DocParser():
    """
    This is an ad-hoc class that handles the responsiblities associated with parsing documents and queries.
    Attributes: stopWords, str array, a list of common English words that don't contribute to the meaning of a text.
    """
    def __init__(self,stopWords:[str])->[str]:                
        '''
        This function initializes the docWords class with a set of stopWords
        Inputs: stopWords, str array, a list of common English words that don't contribute to the meaning of a text.
        Returns: None
        '''
        self.stopWords = self.GetStopWords(stopwords)                
    
    def GetStopWords(self,fileName:str)->[str]:
        '''
        This function takes an input file name that points to a file containing a list of stop words.
        Inputs: fileName, str, the name of a file containing a list of stop words.
        Returns: stopWords, str array, the array of stop words in the input file
        '''
        stopwords_f = open(fileName, 'r') # wk3
        stopWords = stopwords_f.read().split(',')
        stopwords_f.close()
        return stopWords

    def GetWords(self,fileName)->[str]:
        '''
        This function accesses a text file specified by fileName and returns the content in that file.
        Inputs: fileName,str, the name of a file
        Returns: content, str array, an array where each element is a line of text content from the file
        '''
        with open(fileName,"r") as f:
            content = f.readlines()            
            f.close()            
        return content

    def GetElements(self,words:[str],searchkey:str,document:bool)->[str]:
        '''
        This function searches for a particular kind of element body of text in a html/xml document
        Inputs: words, str array, an array of text from a document
        Inputs: searchkey, str, a string search key.
        Inputs: document, bool, a boolean variable that indicates that, 
            if true the incoming words are a document, 
            and if false the incoming words are a query
        Returns: elements, str array, returns an array of text where each element in the array contains the search key.
        '''
        try:
            elements = [w for w in words if searchkey in w][0]
            return elements
        except IndexError:
            if document == True:
                print(f"There was an error: {IndexError} - Maybe the element {searchkey} is not present in your text")
            return None
            
    def GetDocumentID(self,words:[str],document:bool)->str:
        '''
        This function extracts the document ID from a xml document that has some <tag> 
        element that contains a term id="X". This function captures X.
        Inputs: words, str array, a list of the lines of text in an xml document.
        Inputs: document, bool, indicates 
        if true that the words are a document
        if false it indicates the words are a query
        Returns docID, str, the value of the document ID as a string.
        '''
        targetElement = "itemid"
        elementWithItemId = self.GetElements(words,targetElement,document)
        itemIdElement = [w for w in elementWithItemId.split(" ") if targetElement in w][0]
        return itemIdElement.split("=")[1].strip("\"")

    def Remove_Tags(self,text:str)->str:
        '''
        This function removes html tag elements from a string using regex to clear html tags
        Inputs: text,str a string containing xml/html tags to be removed.
        Returns: text,str, a string that no longer contains xml/html tags.
        '''
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('',text)

    def GetEnclosedText(self,words:[str],document:bool)->[str]:
        '''
        This function searches for text wrapped in <text> tags and captures that as a body of content for further processing 
        Inputs: words, str array, a list of the lines of text in an xml document.
        Inputs: document, bool, indicates 
            if true that the words are a document
            if false it indicates the words are a query
        Returns: text_raw,str, the content enclosed in a <text> tag ready for further processing
        '''
        textTag = ['<text>',"</text"]
        textIndices={"start":textTag[0],"end":textTag[1]}

        if document == True:
            try:
                for key,searchkey in textIndices.items():
                    textIndices[key]=words.index(self.GetElements(words,searchkey,document))
                body = words[textIndices["start"]+1:textIndices["end"]]

            except TypeError:
                print(f"There was an error in GetEnclosedText(): {TypeError=}.\nThis is a document: {document}")
                body = words
        else:
            body = words

        body = [self.Remove_Tags(word) for word in body]
        text_raw = "".join(body)       

        return text_raw

    def CleanText(self,raw_text:str)->str:
        '''
        This function takes some input  and removes punctuation, numbers and whitespace and returns the cleaned text
        Inputs: raw_text,str, a string that needs to have punctuation, numbers and white space removed.
        Returns: cleanText, str, a cleaned string
        '''
        # Remove punctuation and numbers
        # Remove stop words
        # Remove tabs,newlines and spaces            
        
        # This is kept for posterity and comparisons sake
        # tokenized_text = word_tokenize(raw_text)
        # print(f"The following is the tokenized text of which there are {len(tokenized_text)} words:\n{tokenized_text}")
        
        stripDigits = lambda x: x.translate(str.maketrans('','',string.digits))
        stripPunctuation = lambda x: x.translate(str.maketrans(string.punctuation," "*len(string.punctuation)))
        stripSpaces = lambda x: re.sub("\s+"," ",x)
        cleanText = stripSpaces(stripPunctuation(stripDigits(raw_text))).strip().split(' ')

        return cleanText
    
    def Parse(self,text:[str],document:bool=True)->DocWords:        
        '''
        Inputs: text, str array, a list of the lines of text in an xml document.
        Inputs: document, bool, indicates 
        if true that the words are a document
        if false it indicates the words are a query
        Returns: doc, Docwords object, a docWords object that contains various properties
        and most importantly the text of a document in the bag of words format.
        '''

        raw_text = self.GetEnclosedText(text,document)        
        clean_text = self.CleanText(raw_text)
        if document==True:
            docID = self.GetDocumentID(text,document)            
            title = self.Remove_Tags(self.GetElements(text,"<title>",document))
        else:            
            docID = "Query"
            title = None
            
        
        wordCount = 0
        terms = {}       
        doc = DocWords(docID,terms,wordCount,title)
        for term in clean_text:
            doc.Plus()          
            doc.addTerm(term,self.stopWords)
                
        return doc
        
    def Parse_doc(self,fileName:str)->DocWords:
        '''
        This function calls the Parse function of the document parser class with some document specific logic.
        Inputs: fileName, str, the name of a file to be parsed into a DocWords object.
        Returns: doc, Docwords object, a docWords object that contains various properties
        and most importantly the text of a document in the bag of words format.
        '''
        words = self.GetWords(fileName)
        doc = self.Parse(words)
        return doc
    
    def Parse_query(self,queryText:str)->dict:    
        '''
        This function calls the Parse function of the document parser class with some query specific logic.
        Inputs: queryText, str, the text to be parsed into a terms dictionary.
        Returns: dictionary object, a term:frequency dictionary representation of the input query.
        '''   
        return self.Parse(queryText,document=False).terms


### Main Functions
# The following two functions service Task 1.1 and 1.2 respectively
# They exist to explicitly satisfy requirements for functions with particular names, inputs and return values.
# In later questions some creative freedom has been employed with the names,inputs and return values
# for the sake of a better design that achieves the same outcomes.

def parse_docs(path:str,stopwords:str)->LinkedList(DocWords):
    '''
    This function takes an input path and an input file name and parses the documents contained in the 
    folder into a collection of DocWords objects.
    Inputs: path, string, the path to a directory containing a collection of xml documents.
    Inputs: stopwords, string, a string that names the stopwords file.
    Returns documents, DocWords LinkedList, An collection of DocWords objects stored in a linked list.
    '''
    parser = DocParser(stopwords)   
    fileNames = glob.glob(path+"*.xml") 
    documents = None

    for fileName in fileNames:
        if documents == None    :
            documents = LinkedList(Node(parser.Parse_doc(fileName)))
        else:
            documents.insert(Node(parser.Parse_doc(fileName)))
    return documents

def parse_query(query:str,stopwords:str)->dict:
    '''
    This function takes an input query and prepares it in the same fashion as a DocWords object yielding a term frequence dictionary.
    Inputs: query, string, a string query.
    Inputs: stopwords, string, a string that names the stopwords file.
    Returns: dictionary, a term frequency dictionary based on the query input
    '''
    parser = DocParser(stopwords)    
    return parser.Parse_query(query)

class DocumentWriter():
    '''
    This class handles the responsibilities of recording information in files.
    Attributes: None
    '''
    def __init__(self)->None:
        '''
        This method instantiates the class.
        Inputs: None
        Returns: None
        '''
        pass

    def WriteDocs(self,documents:LinkedList(DocWords),fileName:str)->None:
        '''
        This function satisfies the requirements of Task 1.3 by writing 
        to file each document ID along with a dictionary of its terms and frequencies.
        Inputs: documents, DocWords LinkedList, the collection of documents to be written to file.
        Inputs: fileName, str, the name of the file to be written to.
        Returns: None
        '''
        self.Remove(fileName)
        
        with open(fileName,"w") as f:
            for doc in documents:
                f.write(str(doc.data))
                f.write(json.dumps(doc.data.DictSortbyFreq(),indent=4))
                f.write("\n")
            f.close()

    def WriteFeatureWeights(self,collectionWeights:dict,fileName:str)-> None:            
        '''
        This function is created to satisfy the requirements of Task 2.3 as it
        writes the tf*idf score of the top 12 ranking terms for each document
        Inputs: collectionWeights, dict, A dictionary of document IDs and dictionaries containing
        terms and tf*idf scores for each corresponding document id.
        Returns: None
        '''
        self.Remove(fileName)
        topN = 11
        with open(fileName,"w") as f:
            for docID,weights in (collectionWeights.items()):
                f.write(f"Document {docID} contains {len(weights)} terms:\n")
                for i,kv in enumerate(weights.items()):
                    if i<=topN:
                        f.write(f"\t{i+1} - {kv[0]}: {kv[1]}\n")
                f.write("\n\n")
        f.close()
        
    def WriteRankings(self,Ranks:dict,query:str,mode:str,fileName:str,returnCount:int)->None:  
        '''
        This function is created to satisfy the requirements of Task 2.3 and 3.3.
        It writes the rank of each document according to some method of ranking for 
        each document against an input query.
        Inputs: Ranks,dict, A dictionary of query, and dictionaries, 
        where the inner dictionary has the docId and the ranking of the corresponding document.
        Inputs: query, str, the string against which the documents are ranked.
        Inputs: mode, str, the method use to calculate the ranking it can be either tfidf or bm25
        Inputs: filesName, str, the name of the file to which the data will be written.
        Inputs: returnCount, int, the number of records to write to file, 12 for Task 2.3 and 5 for Task 3.3
        Returns: None
        '''
        with open(fileName,"a") as f:
            f.write(f"The query is: {query}\n")
            for i,kv in enumerate(Ranks[query].items()):
                if i<=returnCount:
                    f.write(f"\t{i+1} - {kv[0]}-{mode}: {kv[1]}\n")
                    
            f.write("\n\n")
            f.close()
    
    def Remove(self,fileName:str)->None:
        '''
        This function deletes a file from the current working directory to clean up unneeded files.
        Inputs: fileName, str, a string that specifies the name/path of the file to be removed.
        '''
        try: 
            os.remove(fileName)
            print(f"{fileName} was removed")
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            else:
                print(f"The file {fileName} does not exist currently to be removed.")
                pass

class DocumentRanker():
    '''
    The DocumentRanker class services the need of Tasks 2.2, 2.3, 3.2 and 3.3.
    It does this by creating a number of major methods to implement different IR document feature functions
    Tf * Idf and BM25.

    This class takes some liberties with the specific requirements, for instance the function tf*idf doesn't
    return a dictionary of scores for a document, rather, for the sake of efficiency it returns only the scores required for
    the IR model to calculate it's rankings.
    The function is used inside of other functions to generate all required outputs.

    Attributes: documents, DocWords Linked List, a collection of DocWords objects stored as a Linked List.
    Attributes: ndoc, int, the number of documents in the collection
    Attributes: avgdl, int, the average number of words, not terms, in a document
    Attributes: df, dictionary, a term frequency dictionary for the collection where a term 
    is associated with a count showing the number of times that term appears in the collection
    Attributes: collectionFreq, dictionary, a dictionary that counts, for each term, the number of documents that feature each term
    Attributes: writer, Writer Object, a class that handles the writing of data to file.
    '''
    def __init__(self,documents:LinkedList(DocWords),writer:DocumentWriter)->None:
        '''
        This function initialises the ranker class with a collection of DocWords objects and a writer object.
        Inputs: documents, DocWords Linked List, a collection of DocWords objects stored as a Linked List.
        Inputs: writer, Writer Object, a class that handles the writing of data to file.
        Returns: None
        '''
        self.documents = documents
        self.ndocs = len(documents)
        self.avgdl = self.AverageDocLength(documents)
        self.df = self.calc_df(documents)
        self.collectionFreq = self.Get_CollectionFrequencies()
        self.writer = writer

    def AverageDocLength(self,documents:LinkedList(DocWords))-> int:
        '''
        This function satisfies the requirements of Task 3.1 and returns the average length of a document in the collection of DocWords the ranker is working on.
        Inputs: documents, DocWords Linked List, a collection of DocWords objects stored as a Linked List.
        Returns avgdl, float, the average wordCount of a document in the collection
        '''
        return sum(doc.data.wordCount for doc in documents)/len(documents)

    def calc_df(self,documents:LinkedList(DocWords))->dict:
        '''
        This function satisfies the requirements of Task 2.1 which asks for a dictionary object 
        that has, for each term, the total number of occurences in the collection across all documents.
        It creates this by merging dictionaries together with the fancy '|' operator.
        Inputs: documents, DocWords Linked List, a collection of DocWords objects stored as a Linked List.
        Returns: df, dictionary, 
        '''
        collection = {}
        for doc in documents:
            collection = collection|doc.data.terms
        df = {k: v for k, v in sorted(collection.items(),key=lambda term: term[1],reverse=True)}
        return df
        
    def Get_n_k(self,term:str)->int:
        '''
        This function counts the number of documents in the collection of DocWords
        (that the ranker is looking at) that feature the input term.
        Inputs term, string, a tokenized word
        Returns n_k, int, the number of documents that feature that the term occurs at least once in
        '''
        return sum(term in doc.data.terms for doc in self.documents)
    
    def Get_CollectionFrequencies(self)->dict:
        '''
        This function creates the dictionary of terms and how many documents feature those terms.
        Returns: collectionFrequencies, dict, a dictionary of terms and frequencies

        '''
        return {term:self.Get_n_k(term) for term in self.df.keys()}
    
    def CollectionWeights(self,fileName)-> None:
        '''
        This function takes write to file the tf*idf scores for all terms in a document, for all documents.
        This partially satisfies the requirements of Task 2.3 that requests that the top 12 tf*idf terms of each document be written to file.
        Inputs: fileName, str, the name of the file that the collection will be written to.
        Returns: None
        '''
        collectionWeights = {}
        for doc in self.documents:            
            output = {term:self.tfidf(doc,term,freq) for term,freq in doc.data.terms.items()}        
            output = {k:v for k,v in sorted(output.items(),key=lambda term: term[1],reverse=True)}
            collectionWeights[doc.data.docID]= output
            
        self.writer.WriteFeatureWeights(collectionWeights,fileName)

    def tfidf(self,doc:DocWords,key:str,value:int=None)->float:
        '''
        This function implements the tf * idf formula with normalization and smoothing
        it is applied it to a single term in a document and a score is returned.
        Inputs: doc, DocWords object, the document that features the term
        Inputs: key, string, the term that will have it's tf * idf calculated
        Inputs: value, int, an unused input value that helps with the uniformity across score functions
        Returns: tfidf, float, the score of a word from a document.
        '''        
        d = doc.data.terms
        
        numerator = (math.log10(d[key]) + 1) * (math.log(self.ndocs/self.collectionFreq[key]))   
        denominator = [((math.log10(d[x])+1) * math.log(self.ndocs/self.collectionFreq[x]))**2 for x in d.keys()]
        denominator = math.sqrt(sum(denominator))
        
        score = numerator/denominator
        return score

    def bm25(self,doc:DocWords,key:str,value:int=None)->float:
        '''
        This function implements the bm25 formula with document and query weight terms.
        it is applied it to a single term in a document and a score is returned.
        Inputs: doc, DocWords object, the document that features the term
        Inputs: key, string, the term that will have it's tf * idf calculated
        Inputs: value, int, the frequency of the term in the query that has invoked this method.
        Returns: tfidf, float, the score of a word from a document.
        '''
        k1 = 1.2
        k2 = 1000
        b = 0.75
        
        KGen = lambda dl: k1*( (1-b) + (b * dl/self.avgdl) )
        term1 = lambda n: 1/( (n+0.5)/(self.ndocs-n+0.5) )
        term2 = lambda fi,dl: ((k1+1)*fi)/(KGen(dl)+fi)
        term3 = lambda qfi:  ((k2+1)*qfi)/(k2+qfi)         

        score= math.log2(term1(self.collectionFreq[key])) * (term2(doc.data.terms[key],doc.data.wordCount)) * (term3(value))
        return score

    def IRModel(self,query:str,feature:dict,fileName:str,returnCount:int,mode:str)->None:
        '''
        This function implements the doc-at-a-time information retrieval model for either the tf*idf or bm25 
        method of document ranking against an input query.
        Additionally this function writes the rankings to file as required in Tasks 2.3 and 3.3
        Inputs: query, str, this is the input query, against which all documents in the collection will be ranked.
        Inputs: feature, dict, this is a term frequency dictionary resulting from a parsed query.
        Inputs: fileName, str, this is the name of the file to write the results of rankings to.
        Inputs: returnCount, int, this specifies n so that the top-n ranked documents are written to file.
        Inputs: mode, str, this indicates which ranking method is to be used, the two options are "bm25" and "tfidf"
        Returns: None
        '''        
        featureFunction = self.ValidateIRInputs(mode)
        Ranks = {}
        L = {}    
        for term in feature.keys():
            L[term]=[doc for doc in self.documents if term in doc.data.terms.keys()]    
        Rank={}
        for doc in self.documents:
            Rank[doc.data.docID]=0
            for t,l in L.items():            # Get terms and the list of documents that have that term.
                if doc.data.docID in [d.data.docID for d in l if len(l)>0]:
                    Rank[doc.data.docID] += featureFunction(doc,t,feature[t]) * feature[t]
        Ranks[query] = {k:v for k,v in sorted(Rank.items(),key=lambda term: term[1],reverse=True)}
    
        self.writer.WriteRankings(Ranks,query,mode,fileName,returnCount-1)
    
    def ValidateIRInputs(self,mode:str)->callable:
        '''
        This function validates the inputs for the IR model function.
        This function ensures that an error is thrown if an incorrect method is entered and then prompts the user
        to choose instead one of the valid options
        Then this function returns a lambda function to the main method that carries the chosen input method.
        Inputs: mode, str, this indicates which ranking method is to be used, the two options are "bm25" and "tfidf"
        Returns: rankingMethod, lambda function, this returns a lamda function to the IR model that it will use to rank the documents
        '''
        allowableModes = ["bm25","tfidf"]
        if mode not in allowableModes:
            raise Exception(f"Please choose from one of the following allowable modes: {[m for m in allowableModes]}")

        if mode == "bm25":
            return lambda doc,k,v: self.bm25(doc,k,v)
        elif mode =="tfidf":
            return lambda doc,k,v: self.tfidf(doc,k,v)            
            


        


if __name__ == "__main__":
    ### Main Method

    # Set directory to where this file is.
    # The assumption is that a folder called "RCV1v3" 
    # and a file called "common-english-words.txt" will be present
    os.chdir(os.path.dirname(__file__))

    path = "RCV1v3/"
    stopwords = 'common-english-words.txt'

    # This function collects the documents in RCV1v3 and parses them into a collection of docWords objects.
    documents = parse_docs(path,stopwords)

    # The required output files
    fq1 = "Q1.txt"
    fq2 = "Q2.txt"
    fq3 = "Q3.txt"

    # This is a class that handles any file writing tasks listed in the requirements.
    writer = DocumentWriter()

    # Task 1.3 Requires a text file that lists each document ID along with
    #  a dictionary of terms and frequencies
    writer.WriteDocs(documents,fq1)

    # This is a class that handles any information retrieval ranking tasks as outline
    #  in Tasks 2 and 3.
    ranker = DocumentRanker(documents,writer)

    # This method of the ranker class partially satisfies the requirements of Task 2.3
    #  by writing to file the top 12 tfidf scores of terms in each document
    ranker.CollectionWeights(fq2)

    # This element generates some sample queries for Task 2.3,
    #  the requirement states at least 3, we use all documents.
    queryFeatures = {}
    for doc in documents:    
        queryFeatures[doc.data.title]=parse_query(doc.data.title,stopwords)

    # This element performs information retrieval for each query using tfidf,
    #  and writes the top 12 most relevant documents to file
    for query,feature in queryFeatures.items():
        ranker.IRModel(query,feature,fileName=fq2,returnCount=12,mode="tfidf")

    # This element prepares the queries that satisfy the requirements of Task 3.3
    queries = ["This British Fashion","All fashion awards","The stock markets","The British-Fashion Awards"]
    queryStrings = {}
    for doc in queries:
        queryStrings[doc]=parse_query(doc,stopwords)

    # This element performs information retrieval for each query using bm25,
    #  and writes the top 5 most relevant documents to file
    writer.Remove(fq3)
    for query,feature in queryStrings.items():
        ranker.IRModel(query,feature,fileName=fq3,returnCount=5,mode="bm25")


