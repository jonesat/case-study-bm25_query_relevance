Hello!

My name is Andy and thank you for looking at all 724 lines of my code.
My .py file has a main method that has everything setup such that if you run the file by clicking it, all methods will execute and the desired output will be created.
The only requirements are that the common-english-words.txt and folder for RCV1v3 are in the same directory as the .py file.
You can alter the pathes if needed in the first few lines of the if __name__ =="__main__" method.

The document has 4 major classes to note.
1. Linked List (and node) as the primary data structure containing DocWords objects.
2. Document Parser - Has methods to do with the preparation of raw text and translation to a collection of DocWords objects ready for ranking.
3. Document Writer - Has methods to do with the writing of information to files
4. Document Ranker - Has methods to do with the organisation and ranking according to the tf*idf and bm25 methods of ranking documents.
