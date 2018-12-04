import json

def split_review_data(filename, start=0, end=1000):
    review_star1 = ''
    review_star2 = ''
    review_star3 = ''
    review_star4 = ''
    review_star5 = ''

    with open(filename) as f:
        for line in f:
            review = json.loads(line)
            star = int(review['stars'])
            review = json.dumps(review)
            if star == 5:
                review_star5 += review + '\n'
            elif star == 4:
                review_star4 += review + '\n'
            elif star == 3:
                review_star3 += review + '\n'
            elif star == 2:
                review_star2 += review + '\n'
            else:
                review_star1 += review + '\n'

    with open("star1.json", "w") as f:
        f.write(review_star1)
    with open("star2.json", "w") as f:
        f.write(review_star2)
    with open("star3.json", "w") as f:
        f.write(review_star3)
    with open("star4.json", "w") as f:
        f.write(review_star4)
    with open("star5.json", "w") as f:
        f.write(review_star5)


def main():
    split_review_data('yelp_academic_dataset_review.json')



if __name__ == '__main__':
    main()