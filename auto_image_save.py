import requests
import os
from bs4 import BeautifulSoup # pip install beautifulsoup4



# Yahoo image link
image_links = "https://images.search.yahoo.com/search/images;_ylt=Awr9DuWMKTBfHWgA1QdXNyoA;_ylu=X3oDMTB0NjZjZzZhBGNvbG8DZ3ExBHBvcwMxBHZ0aWQDBHNlYwNwaXZz?"

path = input("Enter download path") # define main folder path (Ex: E:\\ImageFolder)

def my_function():
    if not os.path.exists(path):
        os.makedirs(path) # create directory
    download_function() # download image function
   

def download_function():
    category = int(input("How many category pic you want: ")) # means how many you want. Like you wnat image of lion, cat and dog, so it will be 3
   
    query = []
    for q in range(category):
        q = input("Enter Search: ")
        query.append(q) # all search in list
   
   
    n_download = int(input("Enter no of image you want:")) # how many image you want in each folder

    for a in range(len(query)):
        concat_link = image_links + 'q=' + query[a] # url concat
        print(concat_link)
        responses = requests.get(concat_link) # get requests
        text_data = responses.text # text response
   
        soup = BeautifulSoup(text_data, "html.parser") # Python’s html.parser
   
        img_result = soup.findAll("img", {"class":"process"}, limit=n_download) # findAll class: "process"
   
        img_links = []
        for image in img_result:
            download = image["data-src"]
            img_links.append(download) # append all link

        for i, img_link in enumerate(img_links):
            response = requests.get(img_link) # link response
            subpath = path + "/" + query[a] # concat all path with queries
            if not os.path.exists(subpath):
                os.makedirs(subpath) # create sub directory under main folder
           
            edit_name = subpath + "/" + query[a] + str(i+1) + ".jpg" # image name editing
           

            with open(edit_name, "wb") as file:
                file.write(response.content) # save image

if __name__ == '__main__':
    my_function() # main function