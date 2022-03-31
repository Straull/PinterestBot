import requests
import asyncio
from pathlib import Path
import threading


def download(url, queue: asyncio.Queue, lock: threading.Lock, path):
    name = url.split("/")[7].split(".")[0]
    reponse = requests.get(url=url, stream=True)
    with open(f"{path}\{name}.jpg", "wb+") as f:
        reponse.raw.decode_content = True
        f.write(reponse.content)
        f.close()
    if queue.__sizeof__() >= 1:
        try:
            lock.acquire()
            threading.Thread(target=download, args=[queue.get_nowait(), queue, lock, path]).start()
            lock.release()
        except Exception as e:
            if e == asyncio.QueueEmpty:
                pass


def downloadall(urlist):
    queue = asyncio.Queue()
    for url in urlist:
        queue.put_nowait(url)
    path = Path(__file__).resolve().parent
    for x in range(0, 40):
        lock = threading.Lock()
        threading.Thread(target=download, args=[queue.get_nowait(), queue, lock, path]).start()





