import json
# For converting the Yelp JSON data into a more usable format


def convert_yelp(file_name, k=3):
    with open(file_name) as json_data:
        jdata = json.load(json_data)

    # Save a JSON with the restaurant name as key and a list of reviews as the values (maybe not ideal, but it'll do for now)
    rest_data = {}
    review_count = {}
    
    for i, name in jdata['name'].items():
        review = jdata['text'][i]
        if name in rest_data.keys():
            if review_count[name] < k:
                rest_data[name].append(review)
                review_count[name] += 1
        else:
            rest_data[name] = [review]
            review_count[name] = 1
    with open("./data/restaurants_david_%d.json" % k, "w") as fp:
        json.dump(rest_data, fp)

if __name__ == "__main__":
    convert_yelp("./data/restaurants.json", k=3)