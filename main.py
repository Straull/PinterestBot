from py3pin import Pinterest
import os
import imagecolor
import imgdownloaderthread


path = os.getcwd()


theme = input("What is the image theme wanted ? : ")
print("------------First Color------------")
H1 = input("Enter first color Hex Code")
print("------------Second Color------------")
H2 = input("Enter second color hex code")
print("------------Third Color------------")
H3 = input("Enter third color hex code")
colors_list = [H1, H2, H3]

def main():
    client = Pinterest.Pinterest(email='email',
                                 password='password',
                                 username='username',
                                 cred_root='cred_root')

    client.login()

    z = client.search(scope='boards', query=theme)

    urls = []
    for i in z:
        urls.append(i["image_cover_hd_url"])

    imgdownloaderthread.downloadall(urls)

    pathimage = path + "\\images\\"
    if not os.path.exists(pathimage):
        os.makedirs(pathimage)

    imagelist = [f for f in os.listdir(path) if f.endswith(".jpg")]
    for i in imagelist:
        imagecolor.matching_color(i, colors_list, pathimage)

main()