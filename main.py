import os, math, operator, time
from collections import defaultdict, namedtuple
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist, sent_tokenize, word_tokenize
import numpy as np
import re, subprocess, itertools
from tree import TreeParser
from random import shuffle
import pickle
from liblinearutil import train, problem, svm_read_problem, predict
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from collections import Counter
TRAINING_DATA_ROOT = '/home1/c/cis530/hw3/data'
TRAINING_XML_ROOT = '/home1/c/cis530/hw3/xml_data';
TESTING_DATA_ROOT = '/home1/c/cis530/hw3/raw_test_set'
TESTING_XML_ROOT = '/home1/c/cis530/hw3/test_data';

CORENLP_PATH = '/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09';
PARSE_TREE_PATTERN = re.compile('^<parse>(.*?)</parse>$');
WORD_PATTERN = re.compile('^<word>(.*?)</word>$');

def get_all_files(path):
    files_all = PlaintextCorpusReader(path, '.*')
    return files_all.fileids()

def get_full_files(path):
    files = get_all_files(path)
    return [os.path.join(path, f) for f in files]

def load_file_sentences(filename):
    fullstring = open(filename, 'r').read()        
    return [sen.lower() for sen in sent_tokenize(fullstring)] 


def load_file_tokens(filename):
    sentences = load_file_sentences(filename);
    toks = [];
    words= [];
    lmtzr = WordNetLemmatizer()
    st_words=stopwords.words('english')
    invalidChars = set(string.punctuation)
    delimiter = re.compile("[^0-9A-Za-z]+");
    for s in sentences: 
        toks.extend(delimiter.split(s.strip()))
	
#	toks.extend((s.strip()).split())
    for token in toks:
	if len(token) > 0:
		token = token.lower()
#	if token not in st_words and token not in invalidChars: #
#		token=lmtzr.lemmatize(token)                    #
		words.append(token);
	
    return words;	

## LEXICAL FEATURES: 
def extract_single_file_words(xml_file):
    words = [];
    lmtzr = WordNetLemmatizer()
    st_words=stopwords.words('english')
    invalidChars = set(string.punctuation)
    with open(xml_file) as f:
        all_lines = f.readlines();
        for line in all_lines:
            match_obj = WORD_PATTERN.search(line.strip());
            if match_obj:
                token = match_obj.group(1).strip().lower();
#		if token not in st_words and token not in invalidChars:	#
#			token=lmtzr.lemmatize(token)			#
                words.append(token);
    return words;

def extract_top_words(files):
    all_toks = [];
    
    count = 0;
    for f in files:
        toks = extract_single_file_words(f);
        all_toks.extend( toks );
        count = count +1;

    top_words =  sorted( FreqDist(all_toks).items(), key=operator.itemgetter(1), reverse=True )[:10000];
    return [word for word,score in top_words];

def map_unigrams(filename, top_words):
    file_toks = extract_single_file_words(filename);
    freq = FreqDist(file_toks)
    return [int(freq[w]>0) for w in top_words ]


## LEXICAL EXPANSION :
def extract_similarity(top_words):
    vectors = {};
    with open('/project/cis/nlp/tools/word2vec/vectors.txt') as fp:
        for line in fp:
            tokens = line.strip().split(" ");
            if tokens[0] in top_words:
                l = len(tokens);
                vectors[tokens[0]] = [ float(tok) for tok in tokens[1:l] ];

    sim_mat = {};
    vectors_k = vectors.keys();
    for w1 in top_words :
         w1_sim = {}
         for w2 in top_words :
             if w1 != w2 and w1 in top_words and w2 in top_words and w1 in vectors_k and w2 in vectors_k :
                w1_sim[w2] = cosine_similarity(vectors[w1], vectors[w2]);
         if len(w1_sim) > 0:
            sim_mat[w1] = w1_sim;

    return sim_mat;


def map_expanded_unigrams(filename, top_words, similarity_matrix) :
    vec = map_unigrams(filename, top_words);
    non_zero_tokens = [ top_words[ind] for (ind,v) in enumerate(vec) if v == 1];
    keys = similarity_matrix.keys();
    for (ind,tok) in enumerate(vec):
        if tok == 0:
            if top_words[ind] in keys:
                sim_words = similarity_matrix[top_words[ind]];
                l = list(set(sim_words.keys()).intersection(non_zero_tokens));
                if len(l):
                    vec[ind] = max([sim_words[word] for word in l]);
    return vec;

def cosine_similarity(A, B):
    def list_mult(A, B):
        return map(lambda (x,y): x*y, zip(A,B))
    def sum_v(A):
        A2 = [a*a for a in A]
        return sum(A2)
    if (sum(A)*sum(B) == 0): return 0
    return float(sum(list_mult(A,B))) / math.sqrt(float(sum_v(A)*sum_v(B)))



# DEPENDENCIES : 
def extract_single_file_deps(xml_file):
     depname_finder = re.compile('(?<=<dep type=").*(?=">)')
     dep_pairs = []; flag = 0;
     with open(xml_file) as f:
        all_lines = f.readlines();
        for (li, line) in enumerate(all_lines):
             if ("<basic-dependencies>" in line): flag = 1;
             if ("</basic-dependencies>" in line): flag = 0;
             if (flag == 1):
                 m = depname_finder.search(line)
                 if (m != None):
                     depname = m.group(0).strip();
                     governor_line = all_lines[li+1]; dependent_line = all_lines[li+2];
                     ind = governor_line.find("\">"); rind = governor_line.rfind("</gov"); governor_word = governor_line[ind+2:rind];
                     ind = dependent_line.find("\">"); rind = dependent_line.rfind("</dep"); dependent_word = dependent_line[ind+2:rind];
                     triple_tuple = (depname, governor_word.lower(), dependent_word.lower());
                     dep_pairs.append(triple_tuple);
     return dep_pairs

def extract_top_dependencies(files):
     
     token_list = [];
     for f in files:
         token_list.extend(extract_single_file_deps(f))

     top_dep =  sorted( FreqDist(token_list).items(), key=operator.itemgetter(1), reverse=True );
     top_dep = top_dep[:10000];
     return [dep for dep,score in top_dep];

def map_dependencies(xml_file, dep_list):
    onefile_list = extract_single_file_deps(xml_file);
    v = [int(verb in onefile_list) for verb in dep_list];
    return v;

## SYNTACTIC PRODUCTION RULES 
def extract_single_file_prod_rules(xml_file):
    prod_rules = [];
    with open(xml_file) as f:
        all_lines = f.readlines();
        for line in all_lines:
            match_obj = PARSE_TREE_PATTERN.search(line.strip());
            if match_obj:
                parse_tree= match_obj.group(1).strip();
                tree = TreeParser(parse_tree).tree;
                prod_rules.extend(tree.getProdRules());
    return prod_rules;

def extract_prod_rules(files):
    #files = get_all_files(directory)
    token_list = [];
    for f in files:
        token_list.extend(extract_single_file_prod_rules(f));
    top_prod =  sorted( FreqDist(token_list).items(), key=operator.itemgetter(1), reverse=True );
    top_prod = top_prod[:10000];
    return [ rule for rule,score in top_prod];

def map_prod_rules(xml_file, rule_list):
    onefile_list = extract_single_file_prod_rules(xml_file);
    return [int(rule in onefile_list) for rule in rule_list];


def get_mi_weights(bg_corpus, topic_corpus):
    bg_dict = FreqDist(bg_corpus); 
    bg_item_ratio = dict([(w, (bg_dict[w]+0.0)/(len(bg_corpus) + 0.0)) for w in bg_dict])
    topic_dict = FreqDist(topic_corpus);
    topic_item_ratio = dict([(w, (topic_dict[w]+0.0)/(len(topic_corpus) + 0.0)) for w in topic_dict])
    keyitems = [w for w in bg_dict if ((bg_dict[w] >= 5)and(topic_dict.has_key(w)))]
    mi_weight = dict([(w, math.log(topic_item_ratio[w]/(bg_item_ratio[w]))) for w in keyitems])
    return sorted(mi_weight.iteritems(), key=operator.itemgetter(1), reverse=True)

def get_mi_top(bg_corpus, topic_corpus, K):
    ## bg_corpus and topic_corpus are lists of words
    sorted_mi = get_mi_weights(bg_corpus, topic_corpus)
    return [x for x,_ in sorted_mi[:K]]

def mi_feature():
	
	f=open("/home1/c/cis530/project/train_labels.txt","r")
        lines=f.readlines()
        input_data={}

        for line in lines:
                temp_list=line.split()
                input_data[temp_list[0]]=int(temp_list[1])
	
        f =open('train_labels.txt','r')
        files =[]
        for line in f:
                #files.append(path + line.split()[0] + '.xml')
		files.append(line.split()[0]+'.xml')
        #print files
        shuffle(files)
	
	
	#files=get_all_files("/home1/c/cis530/project/train_data")
	path="/home1/c/cis530/project/train_data/"
	
	topic_words1=[]
	topic_words2=[]
	for f in files:
		if input_data[os.path.basename(f)[:-4]]==1:
			topic_words1=topic_words1+load_file_tokens(path+os.path.basename(f)[:-4])
		elif input_data[os.path.basename(f)[:-4]]==-1:
			topic_words2=topic_words2+load_file_tokens(path+os.path.basename(f)[:-4])
		
	mi=get_mi_top(topic_words1+topic_words2, topic_words1, 500)
	mi=mi+get_mi_top(topic_words1+topic_words2, topic_words2, 500)
	
	return mi

def map_mi(filename, mi, flag=False):

	if flag==False:
		path="/home1/c/cis530/project/train_data/"
	elif flag==True:
		path="/home1/c/cis530/project/test_data/"
	#x=set(extract_single_file_words(path+filename+'.xml'))
	x=set(load_file_tokens(path+filename))
	feature=[]
	for item in mi:
		if item in x:
			feature.append(1)
		else:
			feature.append(0)

	return feature	


def extract_mrc_db():
	f = open("/project/cis/nlp/tools/MRC/MRC_parsed/MRC_words",'r')
	mrc_words = [line.rstrip('\n') for line in f]
	return sorted(mrc_words)

def map_mrc_db(xml_file,mrc_words):
	lead_words = extract_single_file_words(xml_file)
	mrc_vec = []
	for word in mrc_words:
		mrc_vec.append(lead_words.count(word)/(len(lead_words)*1.0))
	return mrc_vec


def read_to_dict(filename):
	dic = {}
	lines = open(filename,"r").readlines()
        for line in lines:
                word = line.split()[0]
                value = line.split()[1]
                dic[word] = int(value)
	return dic

def vectorize_word_score(di,lead_words):
	vec = 230*[0]
	#calculate range
	min_word = min(di, key=di.get)
	min_score = di[min_word]
	max_word = max(di, key=di.get)
	max_score = di[max_word]
	r = max_score - min_score
	
	for lead_word in lead_words:
		if lead_word in di:
			score = di[lead_word]
			interval = int(math.ceil((score - min_score)/(r*1.0)))
			vec[interval] += 1
	l = len(lead_words)
	for i in range(0,len(vec)):
		vec[i] =float( vec[i])/l
	return vec
	
def map_word_score(xml_file):
	lead_words = extract_single_file_words(xml_file)
	imag = {}
        fam = {}
        conc = {}
        aoa = {}
        meanc={}
        imag = read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/IMAG")
        fam=  read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/FAM")
        conc =  read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/CONC")
        aoa =  read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/AOA")
        meanc =  read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/MEANC")
 	imag_vec = vectorize_word_score(imag, lead_words)
	fam_vec = vectorize_word_score(fam, lead_words)
	conc_vec = vectorize_word_score(conc, lead_words) 
	aoa_vec = vectorize_word_score(aoa, lead_words) 
	meanc_vec = vectorize_word_score(meanc, lead_words) 			
	return imag_vec + fam_vec + conc_vec + aoa_vec + meanc_vec


# TF. IDF
#get Normalized term frequency of items in list
def get_tf(itemlist):
    tf = Counter(itemlist)
    max_term = max(tf, key=tf.get)
    max_count = tf[max_term]
    for key in tf.keys():
        tf[key] = float (tf[key]) / float (max_count)
    return tf

#get Inverse Document frequency of items in list
def get_idf(lst): ## list of list, each being words in a doc
    df_dict = defaultdict(int)
    df_dict["<unk>"] = 1
    for eachlst in lst:
        lstitems = list(set(eachlst))
        for item in lstitems: df_dict[item] += 1
    N = len(lst)+0.0
    idf_dict = dict((item, math.log(N/v)) for item,v in df_dict.items())
    return idf_dict


#get tokens with top TF.IDF values
def get_tfidf_top(dict1, dict2, k):
    temp = Counter({})
    for tf_key in dict1.keys():
        if tf_key in dict2.keys():
            temp[tf_key] = float(dict1[tf_key]) * float (dict2[tf_key])
	else:
	    temp[tf_key] = float(dict1[tf_key]) * float (dict2['<U N K>'])
    popular_words = sorted(temp, key = temp.get, reverse = True)
    return popular_words[:k]

def extract_top_tfidf():
	label1_tf = []
	label2_tf = []
	all_labels_idf = []
	f=open("/home1/c/cis530/project/train_labels.txt","r")
	lines=f.readlines()
	input_data={}
	
	for line in lines:
		temp_list=line.split()
		input_data[temp_list[0]+'.xml']=int(temp_list[1])
		
	file_labels=input_data
	for file_name in file_labels:
		file_tok = extract_single_file_words("train_xml_data/"+file_name)
		if file_labels[file_name] == 1:
			label1_tf += file_tok
		else:
			label2_tf += file_tok
		all_labels_idf.append(file_tok)
	label1_tf_dict = get_tf(label1_tf)
	label2_tf_dict = get_tf(label2_tf)
	all_labels_idf_dict = get_idf(all_labels_idf)
	label1_tfidf_top = get_tfidf_top(label1_tf_dict,all_labels_idf_dict,20000)
	label2_tfidf_top = get_tfidf_top(label2_tf_dict,all_labels_idf_dict,20000)
	return label1_tfidf_top + label2_tfidf_top


def map_tfidf(xml_file,top_tfidf):
	file_tok = extract_single_file_words(xml_file)
	tfidf_vec = []
	for word in top_tfidf:
		if word in file_tok:
			tfidf_vec.append(1)
		else:
			tfidf_vec.append(1)
	return tfidf_vec
			 


##################################################################################
############################  CLASSIFIER ######################################### 
##################################################################################

def get_label(filename, domain):
    if domain in filename.lower():
        return "1";
    return "-1";

def convert_to_string2( features ) :
    cur_line = "";
    for (fi, f) in enumerate(features):
        if (f==0): continue;
        cur_line = cur_line + " " + str(fi+1) + ":" + str(f);
    return cur_line.strip();

def precision_recall(truth, predicted, num):
    p_truth = 0;
    p_predicted = 0;
    hit = 0;
    for (t,p) in itertools.izip(truth,predicted):
        if t == num :
            p_truth = p_truth + 1;
        if p == num :
            p_predicted = p_predicted + 1;
        if t == p == num:
            hit = hit + 1;

    return (float(hit)/float(p_predicted),float(hit)/float(p_truth))

def run_classifier(train_file, test_file):

        count_one=0

        y_train, x_train = svm_read_problem(train_file)

        counter=0
        while counter<len(y_train):
                if y_train[counter]==-1:
                        count_one=count_one+1
                counter=counter+1

        w1=count_one/float(len(y_train))
        #w1=0.95 # Extra credit
        #w1=0.95 # Extra credit 
        param='-s 0 -w1 '+str(w1)+' -w-1 '+str(1-w1)
        #param='-s 0'   # Extra Credit
        model = train(y_train, x_train, param)

        y_test, x_test = svm_read_problem(test_file)
        p_labels, p_acc, p_vals = predict(y_test, x_test, model, '-b 1')


        accuracy = p_acc[0]

        index=0
        if model.label[0]==1:
                index=0
        elif model.label[1]==1:
                index=1

        counter=0
        prob_list=[]
        while counter<len(p_vals):
                prob_list.append(p_vals[counter][index])
                counter=counter+1

        output_tup=(p_labels, y_test, prob_list)

        return output_tup

#process_corpus creates train and test files in a format that corresponds to liblinear format.

def process_corpus(list_files, filename, top_words, similarity_matrix, dep_list, prod_rules, sub=False, flag=False):

        files=list_files
        
	f4=0
	f5=0
	if sub==False:
#        	f4=open(filename+"4.txt","w")
		f5=open(filename+"5.txt","w")
	elif sub==True:
		f5=open(filename+".txt","w")
		print filename

	f=open("/home1/c/cis530/project/train_labels.txt","r")
	lines=f.readlines()
	input_data={}
	
	for line in lines:
		temp_list=line.split()
		input_data[temp_list[0]+'.xml']=int(temp_list[1])
		
	mi=mi_feature()
	mrc_words=extract_mrc_db()
#	tfidf=extract_top_tfidf()
        count_print=0

        for fil in files:
                list_filename=re.split('/',fil)
                fil_name=list_filename[len(list_filename)-1]
	
		output4=0
		output5=0
		if flag==False:
               		output4=str(input_data.get(fil_name))
		elif flag==True:
			output4='1'
		output5=output4


		vector1=map_unigrams(fil,top_words)
		vector2=map_expanded_unigrams(fil,top_words,similarity_matrix)
		vector3=map_dependencies(fil, dep_list)                
		vector4=map_prod_rules(fil, prod_rules)
		vector7=map_mrc_db(fil,mrc_words)
		vector8=map_word_score(fil)
		#vector9=map_tfidf(fil,tfidf)
		if flag==False:
			vector6=map_mi(fil_name[:-4],mi)
		elif flag==True:
			vector6=map_mi(fil_name[:-4],mi, flag=True)
		vector5=vector1	+vector3+vector4+vector6+vector7+vector8
                counter=0

                #For each file we run all map functions to retreive the corresponding vectors.
                #We then store the nonzero values of those vectors in corresponding files.

#                while counter<len(vector4):

#                        if vector4[counter]!=0:
#                                temp=str(counter+1)+":"+str(vector4[counter])
#                                output4=output4+" "+temp

#                        counter=counter+1

#                output4=output4+"\n"
#                f4.write(output4)
		
		counter=0
		while counter<len(vector5):

                        if vector5[counter]!=0:
                                temp=str(counter+1)+":"+str(vector5[counter])
                                output5=output5+" "+temp

                        counter=counter+1
                output5=output5+"\n"
                f5.write(output5)

#	f4.close()
	f5.close()


def kfold_crossvalidation():
	
#	files=get_full_files('/home1/a/anudeep/comp_ling/project/train')
	path = '/home1/s/shivamda/CL/project/train/'
	f =open('train_labels.txt','r')
	files =[]
	for line in f:
		files.append(path + line.split()[0] + '.xml')
	print files
	shuffle(files)
		
#	chunks=[files[x:x+228] for x in xrange(0, len(files), 228)]	
	
#	chunks[9].append(chunks[10][0])
#	chunks[5].append(chunks[10][1])

#	chunks.remove(chunks[10])
	chunks=[files[x:x+100] for x in xrange(0, len(files), 100)]
	
	f=open("kfold.txt","w")
	pickle.dump(chunks,f)
	f.close()
	
	f=open("kfold.txt","r")
	data=pickle.load(f)
	f.close()

	train_files=[]
	test_files=[]	

	j=0
	while j<10:
		counter=0
		train_files=[]
		test_files=[]
		while counter<10:
			if j==counter:
				print str(j)
				print str(counter)
				test_files=data[counter]	
			else:
				train_files=train_files+data[counter]
			
			counter=counter+1
	
                #Call process_corpus
                prod_rules=extract_prod_rules(train_files)#Call process_corpus
                top_words=extract_top_words(train_files)
                #similarity_matrix=extract_similarity(top_words)
                similarity_matrix={}
                dep_list=extract_top_dependencies(train_files)
                
		process_corpus(train_files, 'train_'+str(j), top_words, similarity_matrix, dep_list, prod_rules)
                process_corpus(test_files, 'test_'+str(j), top_words, similarity_matrix, dep_list, prod_rules)
                j=j+1

	
#kfold_crossvalidation()


def quality(original, predicted):

        length=len(original)
        counter=0
        nr=0
        while counter<length:
                if original[counter]==predicted[counter]:
                        nr=nr+1
                counter = counter +1

        accuracy=float(nr)/length

        return accuracy

def execute_run_classifier():

        i=0
        f=open('results.txt','w')
	sum_acc=0
        while i<10:
                tup=run_classifier('train_'+str(i)+'5.txt', 'test_'+str(i)+'5.txt')
                accuracy=quality(tup[1], tup[0])
		sum_acc+=accuracy
                f.write(str(accuracy)+'\n')
                i=i+1
	print sum_acc/10
	f.write(str(float(sum_acc)/10))
	f.close()
#execute_run_classifier()


def submission_run_classifier(test_files):

        i=0
        f=open('results.txt','w')
        tup=run_classifier('train.txt', 'test.txt')
        p_labels=tup[0]
        while i<len(test_files):
                temp=os.path.basename(test_files[i])
                f.write(str(temp)[:-4]+' '+str(int(p_labels[i]))+'\n')
                i=i+1
        f.close()

def submission():
#        files=get_full_files('/home1/s/shivamda/CL/project/train')
#        shuffle(files)

	path = '/home1/s/shivamda/CL/project/train/'
        f =open('train_labels.txt','r')
        files =[]
        for line in f:
                files.append(path + line.split()[0] + '.xml')
        print files
        shuffle(files)

        train_files=files
	f.close()

#	f = open('test_labels.txt','r')
#	files =[]
#        for line in f:
#                files.append(path + line.split()[0] + '.xml')
#        print files
#	shuffle(files)
	
#	files = files[0:250]
	
        files=get_full_files('/home1/s/shivamda/CL/project/test_xml')
        test_files=files

        prod_rules=extract_prod_rules(train_files)
        top_words=extract_top_words(train_files)
        #similarity_matrix=extract_similarity(top_words)
        similarity_matrix={}
        dep_list=extract_top_dependencies(train_files)

	process_corpus(train_files, 'train', top_words, similarity_matrix, dep_list, prod_rules, sub=True, flag=False)
        process_corpus(test_files, 'test', top_words, similarity_matrix, dep_list, prod_rules, sub=True, flag=True)
                
       	submission_run_classifier(test_files)

submission()
