If you encounter an SSL error:
Make sure that the cert.pem and key.pem files required for HTTPS are in the correct path. If needed, you can generate a temporary certificate by running the following command:

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes
```

Once implemented, you can test the API by opening a browser and entering the following address:
```bash
https://localhost:5000/analyze
```

Using curl on the command line: 

```bash
curl -X POST -F "file=@your_image.dcm" -F "analysis_type=General Description" https://localhost:5000/analyze --insecure
```
---
### Explanation:

* `-X POST` → Send POST request

* `-F "file=@your_image.dcm"` → Upload MRI file (replace your_image.dcm with your own file)

* `-F "analysis_type=General Description"` → Analysis type

* `--insecure` → To ignore SSL in test certificates

### Using Postman:

* Create a new POST request.
* In the URL field, enter this: `https://localhost:5000/analyze`

* Go to the Body tab and select the form-data option. Add two fields:
    * Key file → Type File and select an MRI file.
    * Key analysis_type → Value General Description (or other analysis type)

Click Send.

### Using a Python script:
```python import requests
url = "https://localhost:5000/analyze"
files = {"file": open("your_image.dcm", "rb")}
data = {"analysis_type": "General Description"}
response = requests.post(url, files=files, data=data, verify=False)
print(response.json())
```
Make sure:
* The path to `your_image.dcm` is correct.
* `verify=False` is used to bypass temporary SSL.