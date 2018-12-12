import json
import numpy as np

def split_review_data(filename, total_size=1200):
    outstr = ''
    with open(filename) as f:
        data = f.readlines()
        reviews = [json.loads(x.strip()) for x in data]
        index = list(range(len(reviews)))
        index = np.random.choice(index, size=total_size, replace=False)
        for i in index:
            review = json.dumps(reviews[i])
            outstr += review + '\n'

    with open("small_dataset_1200.json", "w") as f:
        f.write(outstr)


def main():
    split_review_data('yelp_academic_dataset_review.json')



if __name__ == '__main__':
    main()