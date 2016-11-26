#Data Science Capstone Project - Milestone Project
####By John Maged


```{r}
library(tm)
Sys.setenv(JAVA_HOME="C:\\Program Files\\Java\\jre1.8.0_92")
library(rJava)
library(RWeka)
library(googleVis)
library(wordcloud)
```

## Introduction

This milestone report will be applying data science in the area of natural language processing (NLP). 

Below, we will do the following:

* Data Loading,
* Data Cleaning,
* Exploratory Analysis, and
* Conclusion.


Let us start now by data extraction.

## Data Loading

The data set consists of three files in US English.

* en_US.blogs.txt
* en_US.news.txt
* en_US.twitter.txt

## Loading The Dataset 

```{r}
con <- file("en_US.blogs.txt", open="rb")
blogs <- readLines(con, encoding="UTF-8", skipNul=TRUE)
close(con)
rm(con)

con <- file("en_US.twitter.txt", open="rb")
twitter <- readLines(con, encoding="UTF-8", skipNul=TRUE)
close(con)
rm(con)

con <- file("en_US.news.txt", open="rb")
news <- readLines(con, encoding="UTF-8", skipNul=TRUE)
close(con)
rm(con)
```

### Aggreagating A Data Sample

In order to enable faster data processing, a data sample from all three sources was generated.

```{r, results='hide', message=FALSE, warning=FALSE, eval=TRUE, echo=TRUE}
sampleTwitter <- twitter[sample(1:length(twitter),5000)]
sampleNews <- news[sample(1:length(news),5000)]
sampleBlogs <- blogs[sample(1:length(blogs),5000)]
textSample <- c(sampleTwitter,sampleNews,sampleBlogs)

## Saving sample..
writeLines(textSample, "textSample.txt")

## Reading from the sample file..
theSampleCon <- file("textSample2.txt")
theSample <- readLines(theSampleCon)
close(theSampleCon)
```

## Summary Statistics

```{r}
# Checking the size and length of the files and calculate the word count
blogsFile <- file.info("en_US.blogs.txt")$size
newsFile <- file.info("en_US.news.txt")$size
twitterFile <- file.info("en_US.twitter.txt")$size 
sampleFile <- file.info("textSample.txt")$size

# Line counts of different files
blogsLength <- length(blogs)
newsLength <- length(news)
twitterLength <- length(twitter)
sampleLength <- length(textSample)

# Word counts
blogsWords <- sum(sapply(gregexpr("\\S+", blogs), length))
newsWords <- sum(sapply(gregexpr("\\S+", news), length))
twitterWords <- sum(sapply(gregexpr("\\S+", twitter), length))
sampleWords <- sum(sapply(gregexpr("\\S+", textSample), length))
```

**Building the data frame.**

```{r}
# Vectors of line count, word count, and document size for each file.
lineCounts <- c(twitterLength, newsLength, blogsLength, sampleLength)
names(lineCounts)<-c("twitter","news","blogs","sample")
wordCounts <- c(twitterWords, newsWords, blogsWords, sampleWords)
names(wordCounts)<-c("twitter","news","blogs","sample")
sizes<- c(twitterFile,newsFile,blogsFile,sampleFile)
names(sizes)<-c("twitter","news","blogs","sample")

# Building the data frame..
summary.df <- as.data.frame(cbind(lineCounts,wordCounts,sizes))
rownames(summary.df)<-c("twitter","news","blogs","sample")
colnames(summary.df)<-c("Line Counts", "Word Counts", "Document Size")
```

The following table provides an overview of the imported data. You will find the number of lines in each file including the three files plus the sample file. In addition, you will find the number of words and size of each file.  

```{r}
summary.df
```
```
##         Line Counts Word Counts Document Size
## twitter     2360148    30373583     167105338
## news        1010242    34372530     205811889
## blogs        899288    37334131     210160014
## sample        15000      446178       2559254
```

**Here are a plotting of the above summary:**

```{r, eval=TRUE, echo=FALSE}
barplot(t(as.matrix(summary.df)), horiz = T, col=c("red","green","blue"),legend = rownames(t(as.matrix(summary.df))),beside=TRUE)
```

## Building a clean corpus

**Using the tm package, corpus is cleaned** - for example, text data is converted into lower case, further punction, numbers and URLs are getting removed. Next to that stop words are erased from the text sample. At the end we are getting a clean text corpus which enables an easy subsequent processing.


```{r, results='hide', message=FALSE, warning=FALSE, eval=TRUE, echo=TRUE}
theSampleCon <- file("textSample.txt")
textSample <- readLines(theSampleCon)
close(theSampleCon)

## Build the corpus, and specify the source to be character vectors 
cleanSample <- Corpus(VectorSource(textSample))
rm(textSample)

## Make it work with the new tm package
cleanSample <- tm_map(cleanSample,
                      content_transformer(function(x) 
                              iconv(x, to="UTF-8", sub="byte")))

## Convert to lower case
cleanSample <- tm_map(cleanSample, content_transformer(tolower), lazy = TRUE)

## remove punction, numbers, URLs, stop, and stemming
cleanSample <- tm_map(cleanSample, content_transformer(removePunctuation))
cleanSample <- tm_map(cleanSample, content_transformer(removeNumbers))
removeURL <- function(x) gsub("http[[:alnum:]]*", "", x) 
cleanSample <- tm_map(cleanSample, content_transformer(removeURL))
cleanSample <- tm_map(cleanSample, stripWhitespace)
cleanSample <- tm_map(cleanSample, removeWords, stopwords("english"))
cleanSample <- tm_map(cleanSample, stemDocument)
cleanSample <- tm_map(cleanSample, stripWhitespace)
```

## Exploratory analysis 

### The N-Gram Tokenization

In Natural Language Processing (NLP), an n-gram is a contiguous sequence of n items from a given sequence of text or speech.

The following function is used to extract 1-grams, 2-grams and 2-grams from the cleaned text corpus.


```{r, results='hide', message=FALSE, warning=FALSE, eval=TRUE, echo=TRUE}
## Budilding the n-grams
finalCorpusDF <-data.frame(text=unlist(sapply(cleanSample,`[`, "content")), 
                           stringsAsFactors = FALSE)


## Building the tokenization function for the n-grams
ngramTokenizer <- function(theCorpus, ngramCount) {
        ngramFunction <- NGramTokenizer(theCorpus, 
                                Weka_control(min = ngramCount, max = ngramCount, 
                                delimiters = " \\r\\n\\t.,;:\"()?!"))
        ngramFunction <- data.frame(table(ngramFunction))
        ngramFunction <- ngramFunction[order(ngramFunction$Freq, 
                                             decreasing = TRUE),][1:10,]
        colnames(ngramFunction) <- c("String","Count")
        ngramFunction
}

```

By the usage of the tokenizer function for the n-grams a distribution of the following top 10 words and word combinations can be inspected. Unigrams are single words, while bigrams are two word combinations and trigrams are three word combinations.

###Top Unigrams
```{r}
## urigram plot
unigram <- ngramTokenizer(finalCorpusDF, 1)
print(unigram)
```
```
##       String Count
## 20516   said  1439
## 26302   will  1434
## 17019    one  1353
## 13858   like  1205
## 24034   time  1152
## 9759     get  1144
## 12778   just  1104
## 3900     can  1022
## 26790   year   982
## 9961      go   981
```


###Top Bigrams
```{r}
## bigram plot
bigram <- ngramTokenizer(finalCorpusDF, 2)
print(bigram)
```

```
##             String Count
## 100506   last year    92
## 154766   right now    88
## 123611    new york    82
## 100498   last week    79
## 210025    year ago    77
## 107491   look like    70
## 64790    feel like    65
## 67195   first time    61
## 84353  high school    57
## 52209    dont know    52
```


###Top Trigrams
```{r}
## trigram plot
trigram <- ngramTokenizer(finalCorpusDF, 3)
print(trigram)
```

```
##                     String Count
## 168904      rain rain rain    26
## 222527        two year ago    15
## 141788       new york citi    12
## 163384 presid barack obama    10
## 92430     happi mother day     9
## 92433       happi new year     8
## 141832       new york time     8
## 31885        cant wait see     7
## 67094      everi singl day     7
## 189131      sever year ago     7
```



## Creating a wordcloud

A word cloud usually provides a first overview of the word frequencies. The word cloud displays the data of the aggregated sample file.

```{r, message=FALSE, warning=FALSE, echo=FALSE}
wordcloud(cleanSample, max.words = 200, random.order = TRUE)
```

# Conclusion

* A data sample file was created as the raw data size is big and takes a lot of time in processing. This was very useful when working on this sample on the next steps - data cleaning and exploratory analysis.I will work on how sampling is done in a better way that will not impact the final results.

* In Data Cleaning, I converted all text into lower case and removed any punctuations, numbers, URLs, stop words, and did stemming.

* In exploratory analysis, I did some analysis for N-grams as you saw earlier in this report. Top unigrams, bigrams, and trigrams in the sample data file was represented.

# The Following Steps

**The next steps in the project will be:**

* Enhancing the sampling process on the raw data or find a way enhancing the performance when applying different processing tasks on the raw data itself.

* Building the prediction model that will predict the next word a user wants to write as the SwiftKey applications.
