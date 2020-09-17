import requests
import json

url="https://us-central1-mlservice-manning.cloudfunctions.net/myCloudEcho"

### url="http://httpbin.org/headers"

myheaders={'message':'yowza', 'name':'wibbler', 'id':'1245'}
### r=requests.get(url, headers=myheaders)

headers = {'content-type': 'application/json'}

r = requests.post(url, data=json.dumps(myheaders), headers=headers)

print(f"Status  : {r.status_code}" )
print(f"Text    : {r.text}" )
print(f"Headers :")
for k,v in r.headers.items() :
    print(f"\t{k} : {v}")

### print(f"JSON   : {r.json()}" )
