## Review Auto Completion

### Introduction

Inspired by *Smart Compose*, a new feature recently launched by Google for its email service that provides automatic sentence completion using machine learning and artificial intelligence techniques, we aim to build a system that could do auto sentence completion on yelp user reviews.

<!--
##### Team members: 
Haiyang Huang (hyhuang), Xinglian Liu (xinglian), Yuanhang Luo (royluo), Yuzhou Mao (myz)
-->

### Dataset

  * The dataset we use is the [Yelp Open Dataset](https://www.yelp.com/dataset). The Yelp Open Dataset consists of a few JSON data files (business.json, review.json, user.json, etc). We only use one of the JSON data files in the dataset, namely review.json. 

  * In review.json, there is one JSON object per line. One sample json object in review.json is provided as follows.

```
{
    // string, 22 character unique review id
    "review_id": "zdSx_SD6obEhz9VrW9uAWA",

    // string, 22 character unique user id, maps to the user in user.json
    "user_id": "Ha3iJu77CxlrFm-vQRs_8g",

    // string, 22 character business id, maps to business in business.json
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // integer, star rating
    "stars": 4,

    // string, date formatted YYYY-MM-DD
    "date": "2016-03-09",

    // string, the review itself
    "text": "Great place to hang out after work: the prices are decent, and the ambience is fun. It's a bit loud, but very lively. The staff is friendly, and the food is good. They have a good selection of drinks.",

    // integer, number of useful votes received
    "useful": 0,

    // integer, number of funny votes received
    "funny": 0,

    // integer, number of cool votes received
    "cool": 0
}
```

### Requirements

  * python (3.5.x, 3.6.x)
  
    * tensorflow==1.12.0
    
    * gensim==3.6.0
    
    * nltk==3.4
    
    * pygtrie==2.3

### Usage

  * The complete documentation of the code is at the wiki page: https://github.com/liuxinglian/autocomplete/wiki.

  * Usage of the models that we built is as follows:
  
    * Model 1
    
      ```shell
      python model1/model1.py
      ```
  
    * Model 2
  
      ```shell
      python model2/model2.py
      ```

    * Model 3

      ```shell
      python model3/model3_star.py
      ```
