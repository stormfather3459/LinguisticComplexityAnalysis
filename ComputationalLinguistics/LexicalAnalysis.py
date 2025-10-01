import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import FreqDist
from nltk.chunk import RegexpParser
import numpy as np
import re 
import statistics
from nltk.parse.chart import ChartParser
from nltk import Tree
import json
import pprint
import string
from collections import Counter

nltk.download('punkt')
nltk.download('atis')  # Download the ATIS grammar


def load_csv(path):
    
    df = pd.read_csv(path)
    #list column names
   # print(df.columns.values)

    return df


def get_freq_dist_per_df(df, text_column='TOPIC'):
    all_tags = []
    for text in df[text_column].dropna():
        tokens = word_tokenize(str(text))
        tagged_tokens = pos_tag(tokens)
        all_tags.extend([tag for (word, tag) in tagged_tokens])

    fd = FreqDist(all_tags)
    return fd

def main():
    adminDF = load_csv("ADMIN.csv")
    dubDF = load_csv("DU.csv")   



    #print(shortenedDF.shape)
    adminDF.sort_values(by = ['PID'], inplace = True)
    adminText = ''.join(adminDF['TOPIC'].tolist())
    
    
    dubDF.sort_values(by = ['pid'], inplace = True)
    clean_list = [str(item) if item is not None else "" for item in dubDF['contribution'].tolist()]
    dubText = ''.join(clean_list)
    #dubText = ''.join(dubDF['contribution'].tolist())

    print("Administration Processing")
    lex_compex = Lexical_Complexity(adminText)
    lex_compex.printStats()
    print("---------------------------------------------------------------------")
    print("---------------------------------------------------------------------")
    print("---------------------------------------------------------------------")
    str_compex = Structural_Complexity(adminText)
    str_compex.printStats()
    
    print("Dublin Processing")
    lex_compex = Lexical_Complexity(dubText)
    lex_compex.printStats()
    print("---------------------------------------------------------------------")
    print("---------------------------------------------------------------------")
    print("---------------------------------------------------------------------")
    str_compex = Structural_Complexity(dubText)
    str_compex.printStats()

   # print(adminDist.most_common())
  #  textExample = adminPanel['TOPIC'].iloc[0]
   # fd = freqDist(textExample)
   # print(fd.most_common())


class Lexical_Complexity:
    def __init__(self, text):
        self.textInput = text
        self.wordAmount = 0
        self.word_freq = {}

        self.dict_fd = {}
        self.dict = {
            "word_length" : {},
            "word_frequencies" : {},
            "word_categories" : {},
            "letter_frequencies" : {}
        }
        self.count_word_frequencies()
        self.freqDist()
        self.analyze_word_lengths()
        self.count_letter_frequencies()
        self.returnStats()

    def freqDist(self):
        tokens = word_tokenize(self.textInput)
        tagged_tokens = pos_tag(tokens)
        fd = nltk.FreqDist(tag for(word, tag) in tagged_tokens)
        maxItem = fd.max()
        listOfValues = list(fd.values())
        minValue = min(listOfValues)
        maxValue = str(max(listOfValues))
        mean = np.mean(listOfValues)
        median = np.median(listOfValues)
        std_dev = np.std(listOfValues)
        dictio = {
            "categories" : fd.most_common(),
            "count_of_categories": fd.B(),
            'mean_frequency':  mean,
            'max_sample': maxItem,
            "max" : maxValue,
            'min': minValue,
            'standard_deviation' : std_dev,
            'median': median
        }
        self.dict['word_categories'] = dictio
       # self.dict[3] = dictio

    def count_letter_frequencies(self):
        text = self.textInput
        lowerCaseText = text.lower()
        lowerCaseLetters = [char for char in lowerCaseText if char in string.ascii_lowercase]
        letter_counts = Counter(lowerCaseLetters)
        sorted_letter_counts = {k: letter_counts[k] for k in sorted(letter_counts.keys())}

        letterDict = {
            "count": sorted_letter_counts,
            "max": letter_counts.most_common()[0],
            "min": letter_counts.most_common()[-1],
            "mean": np.mean(list(letter_counts.values())),
            "median":np.median(list(letter_counts.values())),
            "standard-deviation": np.std(list(letter_counts.values()))
        }
        self.dict['letter_frequencies'] = letterDict
        #self.dict[4] = letterDict
        
    def count_word_frequencies(self, new_text=None):
        self.word_freq.clear()
        if new_text:
            self.textInput = new_text
        
        # remove punct, make all lowercase
        cleaned_text = re.sub(r'[^\w\s]', '', self.textInput.lower())
        
        # Split into words
        words = cleaned_text.split()
        self.wordAmount = len(words)
        
        # Count the frequency of each word
        for word in words:
            if word in self.word_freq:
                self.word_freq[word] += 1
            else:
                self.word_freq[word] = 1
        
        #sort by frequency
        #self.word_freq = dict(sorted(self.word_freq.items(), key=lambda item: item[1], reverse=True))

        #sort by word length
        sorted_freq  = dict(sorted(self.word_freq.items(), key=lambda item: len(item[0]), reverse=True))

        freq_values = np.array(list(sorted_freq.values()))
            
            # Calculate frequency distribution metrics
        self.dict["word_frequencies"] = {
            'total_words': self.wordAmount,
            'unique_words': len(sorted_freq),
            'frequency_distribution': {
                'min_frequency': int(np.min(freq_values)),
                'max_frequency': int(np.max(freq_values)),
                'mean_frequency': round(float(np.mean(freq_values)), 2),
                'median_frequency': round(float(np.median(freq_values)), 2),
                'std_frequency': round(float(np.std(freq_values)), 2)
            },
            'top_words': dict(list(sorted_freq.items())[:10])  # Top 10 most frequent words
        }

    def analyze_word_lengths(self):
        if not self.word_freq:
            return None
        
        word_lengths = [len(word) for word in self.word_freq.keys()]
        avg_length = sum(word_lengths) / len(word_lengths)
        max_length = max(word_lengths)
        min_length = min(word_lengths)
        
        max_count = sum(1 for word in self.word_freq if len(word) == max_length)
        min_count = sum(1 for word in self.word_freq if len(word) == min_length)
   
       
        median = np.median(word_lengths)
        std_dev =  np.std(word_lengths)
        
        self.dict["word_length"] =  {
            'mean_length': avg_length,
            'max_length': max_length,
            'min_length': min_length,
            'max_count': max_count,
            'min_count': min_count,
            'standard_deviation' : std_dev,
            'median': median

        }
        
    def print(self):
        print(f"word counts: {self.word_lengths}")
        print(f"Frequency Distrubition: {self.dict_fd.most_common()} ")
        print("Amount of unique words: " + str(len(self.word_freq)))
        print("Count of all words: " + str(self.wordAmount))
        for word, freq in self.word_freq.items():
           print(f"{word}: freq - {freq} | len - {len(word)}")

    def printStats(self):
       # for name, data in self.dict.items():
       #     print("---------------------------------")
       #     print(name)
       #     print("")
       #     print(data)
       #
       #    # print(self.dict[i+1].values())
       #     print("---------------------------------")

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.dict)


      #  for dataField, dataEntry in self.dict.items():
      #      print("---------------------------------")
      #      print(dataField)
      #      print("---------------------------------")
      #      for key in dataEntry:
      #          print(key + ':', dataEntry[key])
    
    def returnStats(self):
        return self.dict


class Structural_Complexity:

    def __init__(self, text):
        self.dict = {
        "sentence_length" : {},
        "speech_distribution" : {},

        "noun_phrase_compl" : {},
        "conjunctions" : {}
        }
        self.parse_trees =[]
        self.savedText = text
        self.sentences = nltk.sent_tokenize(self.savedText)
        self.tokenized_sentences = [nltk.word_tokenize(sent) for sent in self.sentences]
        self.sentence_data()
        self.speechDistribution()
       # self.treeDepthCal()
        self.clauseCal()
        #self.clauseComplexity()
        self.np_vp_ration()

    def sentence_data(self):

        sent_dict = {
            "mean" : 0,
            "max" : 0,
            "min" : 0,
            "standard-deviation" : 0,
            "median" : 0
        }

        sentence_lengths = [len(sentence) for sentence in self.tokenized_sentences]
        sent_dict["mean"] = np.mean(sentence_lengths)
        sent_dict["max"] = np.max(sentence_lengths)
        sent_dict["min"] = np.min(sentence_lengths)
        sent_dict["standard-deviation"] = np.std(sentence_lengths)
        sent_dict["median"] = np.median(sentence_lengths)
        self.dict["sentence_length"] = sent_dict
    
    def speechDistribution(self):
        pos_tags = [nltk.pos_tag(sent) for sent in self.tokenized_sentences]
        pos_distribution = {}
        for sent_tags in pos_tags:
            for _, tag in sent_tags:
                pos_distribution[tag] = pos_distribution.get(tag, 0) + 1

       # self.dict["speech_distribution"] = pos_distribution

        pos_stats = {
       #     "tag_values" : [""],
        #    "tag_frequencies" : 0,
            "tag_values": 0,
            "mean": 0,
            "max": 0,
            "min": 0,
            "standard-deviation": 0,
            "median": 0
        }
      #  tag_values = list(pos_distribution.keys())
        tag_frequencies = list(pos_distribution.values())

       # pos_stats["tag_values"] = tag_values
       # pos_stats["tag_frequencies"] = tag_frequencies
        pos_stats["tag_values"] = list(pos_distribution.items())
        pos_stats["mean"] = np.mean(tag_frequencies)
        pos_stats["max"] = np.max(tag_frequencies)
        pos_stats["min"] = np.min(tag_frequencies)
        pos_stats["standard-deviation"] = np.std(tag_frequencies)
        pos_stats["median"] = np.median(tag_frequencies)
        
        self.dict["speech_distribution"] = pos_stats

    def treeDepthCal(self):
        
        tree_depth_stats = {
            "mean" : 0,
            "median" : 0,
            "max" : 0 ,
            "min" : 0 ,
            "standard-deviation" : 0 
        }
        # Generate parse trees
        
        for sent in self.tokenized_sentences:
            try:
                tagged_sent = nltk.pos_tag(sent)
                parse_tree = nltk.ne_chunk(tagged_sent)
                self.parse_trees.append(parse_tree)
            except Exception as e:
                print(f"Error parsing sentence: {sent}")
                print(e)
        
        # Calculate tree depth method
        def tree_depth(tree):
           # print("START--------------------------")
            #print(tree)
            #print("END--------------------------")
            return np.sum(tree.leaves)


            if isinstance(tree, nltk.Tree):
                if len(tree) == 0:
                    return 1
                return 1 + max(tree_depth(t) for t in tree)
            return 0

        tree_depths = [tree_depth(tree) for tree in self.parse_trees]
        tree_depth_stats["mean"] = np.mean(tree_depths)
        tree_depth_stats["median"] = np.median(tree_depths)
        tree_depth_stats["max"] = np.max(tree_depths)
        tree_depth_stats["min"] = np.min(tree_depths)
        tree_depth_stats["standard-deviation"] = np.std(tree_depths)
        self.dict["tree_depth"] = tree_depth_stats
        return
    
    def clauseComplexity(self):
        try:
            atis_grammar = nltk.data.load('grammars/large_grammars/atis.cfg')
        except:
            print("failed to load grammar")

        def count_clauses(tree):
            count = 0
            if isinstance(tree, Tree):
                if tree.label() in ('S', 'SQ', 'SBAR', 'SINV'):
                    count += 1
        
                for child in tree:
                    count += count_clauses(child)
            return count
        
        
        # Create a parser with the ATIS grammar
        
        parser = ChartParser(atis_grammar)
        sentence_clause_counts = []
        for sentence in self.sentences:
            tokens = nltk.word_tokenize(sentence.lower())
            try:
                # Try to parse the sentence
                trees = list(parser.parse(tokens))
                
                if trees:
                    # Count clauses in the first parse tree
                    sentence_clause_counts.append(count_clauses(trees[0]))
                    
                    # Return results
                    return {
                        "sentence": sentence,
                        "clause_count": clause_count,
                        "parse_success": True
                    }
                else:
                    return {
                        "sentence": sentence,
                        "parse_success": False,
                        "message": "Parsing failed with ATIS grammar"
                    }
            except Exception as e:
                # Handle exceptions
                return {
                    "sentence": sentence,
                    "parse_success": False,
                    "error": str(e),
                }
        clause_compl = {
            "sum": 0,
            "mean" : 0,
            "median" : 0,
            "max" : 0 ,
            "min" : 0 ,
            "standard-deviation" : 0 
        }
        clause_compl["sum"] = np.sum(sentence_clause_counts)
        clause_compl["mean"] = np.mean(sentence_clause_counts)
        clause_compl["median"] = np.median(sentence_clause_counts)
        clause_compl["max"] = np.max(sentence_clause_counts)
        clause_compl["min"] = np.min(sentence_clause_counts)
        clause_compl["standard-deviation"] = np.std(sentence_clause_counts)
        self.dict['clause_complexity'] = clause_compl
        return

    def np_vp_ration(self):

        #define base np and vp grammar patterns for nltk
        np_grammar = r"""
            NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
                {<PRP>}
        """
    
        vp_grammar = r"""
            VP: {<MD>?<VB.*>+<RB.*>*}
                {<TO><VB><RB.*>*}"
        """

        np_parser = RegexpParser(np_grammar)
        vp_parser = RegexpParser(vp_grammar)
       # 
        sentences = self.tokenized_sentences
        tagged_sentences = [pos_tag(sentence) for sentence in sentences]
        np_count = 0
        for tagged_sent in tagged_sentences:
            tree = np_parser.parse(tagged_sent)
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    np_count += 1
        

        vp_count = 0
        for tagged_sent in tagged_sentences:
            tree = vp_parser.parse(tagged_sent)
            for subtree in tree.subtrees():
                if subtree.label() == 'VP':
                    vp_count += 1
        ratio = 0
        if vp_count > 0:
            ratio = np_count / vp_count
        else:
            print("division by zero")

        complexity = {
            "np_count" : np_count,
            "vp_count" : vp_count,
            "ratio_np_vp" : ratio 
        }
        self.dict['noun_phrase_compl'] = complexity
        return

    def clauseCal(self):
        subordinating_conjunctions = {
            'after','although','as','as if','as long as','as much as','as soon as',
            'as though','because','before','even','even if','even though','if','if only',
            'if when','if then','inasmuch','in order that','just as','lest','now','now since',
            'now that','now when','once','provided','provided that','rather than',
            'since','so that','supposing','than','that','though','till','unless','until',
            'when','whenever','whereas','where if','wherever','whether','while'
        }

        sentences = self.tokenized_sentences
        subordinating_conjunctions_counts = []
        for sentence in sentences:
            tagged_words = nltk.pos_tag(sentence)
            conj_count = sum(1 for word, tag in tagged_words if tag == 'CC' or word.lower() in subordinating_conjunctions)
            subordinating_conjunctions_counts.append(conj_count)

        clauseCalDict = {
            "mean" : 0,
            "median" : 0,
            "max" : 0 ,
            "min" : 0 ,
            "standard-deviation" : 0 }
        
        clauseCalDict["mean"] = np.mean(subordinating_conjunctions_counts)
        clauseCalDict["median"] = np.median(subordinating_conjunctions_counts)
        clauseCalDict["max"] = np.max(subordinating_conjunctions_counts)
        clauseCalDict["min"] = np.min(subordinating_conjunctions_counts)
        clauseCalDict["standard-deviation"] = np.std(subordinating_conjunctions_counts)
        self.dict['conjunctions'] = clauseCalDict
        return 
    
    def printStats(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.dict)

    
    def saveStats(self):
        return self.dict




if __name__ == "__main__":
    main()