import requests
import time
def translate_eng(x):
    time.sleep(2)
    return requests.get(url="http://139.199.209.106/trans/google.action?from=en&to=zh&query="+x).text