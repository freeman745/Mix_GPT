import requests
import os
import sys


def upload(roo_dir):
    for folder_name, subfolders, filenames in os.walk(roo_dir):
        for filename in filenames:
            upload_file = os.path.join(folder_name, filename)
            files_t = {'file': (filename, open(upload_file, 'rb'))}
            headers = {'File-Name': filename}
            r = requests.post("http://127.0.0.1:5000/upload", files=files_t, headers=headers)

            print(r.text)


def sendurl(url):
    data = {'url': url}
    r = requests.post("http://127.0.0.1:5000/savehtml", data=data)

    print(r.text)


def delete(filename):
    data = {'filename': filename}
    r = requests.post("http://127.0.0.1:5000/delete", data=data)

    print(r.text)


def learn():
    r = requests.post("http://127.0.0.1:5000/learn")

    print(r)


def chat(message, method):
    #r = requests.post("http://127.0.0.1:5000/predict", data={'message': message, 'stream':int(method)})
    r = requests.post("http://hgpt.openweb3.ai/predict", data={'message': message, 'stream':int(method)})
    if int(method) == 0:
        print(r.text)
    else:
        decoder = r.iter_content(chunk_size=1024)

        if r.status_code == 200:
            for chunk in decoder:
                text = chunk.decode("utf-8")
                print(text)
                print(f"Received {len(chunk)} bytes")
            #print(response_text)
        else:
            print("请求失败:", r.status_code)


def dbsearch(message):
    r = requests.post("http://127.0.0.1:5000/dbsearch", data={'message': message})

    print(r.text)


function_name = sys.argv[1]
method = sys.argv[2]

if function_name == 'upload':
    upload('test_folder/')
elif function_name == 'delete':
    delete('test0.pdf')
elif function_name == 'learn':
    learn()
elif function_name == 'chat':
    message = "介绍下东京"
    chat(message,method)
elif function_name == 'saveurl':
    sendurl('https://blog.csdn.net/john_bian/article/details/71025372')
elif function_name == 'dbsearch':
    message = "openweb3"
    dbsearch(message)
else:
    print("Invalid function name. Only support: upload, learn, chat, delete, saveurl, dbsearch.")
