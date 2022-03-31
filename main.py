from py3pin import Pinterest
import os
import imagecolor
import imgdownloaderthread

path = os.getcwd()

theme = input("Quel est le theme d'image que vous souhaitez telecharger ? : ")
print("------------Premiere Couleur------------")
R1 = eval(input("Entrer la valeur de Rouge souhaite : "))
V1 = eval(input("Entrer la valeur de Vert souhaite : "))
B1 = eval(input("Entrer la valeur de Bleu souhaite : "))
print("------------Deuxieme Couleur------------")
R2 = eval(input("Entrer la valeur de Rouge souhaite : "))
V2 = eval(input("Entrer la valeur de Vert souhaite : "))
B2 = eval(input("Entrer la valeur de Bleu souhaite : "))
print("------------Troisieme Couleur------------")
R3 = eval(input("Entrer la valeur de Rouge souhaite : "))
V3 = eval(input("Entrer la valeur de Vert souhaite : "))
B3 = eval(input("Entrer la valeur de Bleu souhaite : "))
RVB1 = (R1, V1, B1)
RVB2 = (R2, V2, B2)
RVB3 = (R3, V3, B3)
colors_list = [RVB1, RVB2, RVB3]


def main():
    client = Pinterest.Pinterest(email='email',
                                 password='password',
                                 username='username',
                                 cred_root='cred_root')

    client.login()

    z = client.search(scope='boards', query=theme)
    urls = []
    name = []
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



