#!/usr/bin/env bash

#download data
wget http://www.statmt.org/wmt11/training-monolingual-news-2011.tgz \
    && tar xvzf training-monolingual-news-2011.tgz \
    && rm training-monolingual-news-2011.tgz

#download postprocess script
wget http://www.statmt.org/wmt08/scripts.tgz \
    && tar xvzf scripts.tgz \
    && rm scripts.tgz

#apply postprocess scripts
./scripts/tokenizer.perl -l en <./training-monolingual/news.2011.en.shuffled> ./training-monolingual/news.2011.en.shuffled.tokenized
cat ./training-monolingual/news.2011.en.shuffled.tokenized | ./scripts/lowercase.perl > ./training-monolingual/news.2011.en.shuffled.tokenized.lowercased

rm ./training-monolingual/news.2011.en.shuffled.tokenized

# remove duplicates

sort -u ./training-monolingual/news.2011.en.shuffled.tokenized.lowercased > ./training-monolingual/news.2011.en.shuffled.tokenized.lowercased.unique
