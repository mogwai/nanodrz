import requests
import tarfile

def download_speech():
        link = "https://openslr.elda.org/resources/60/test-clean.tar.gz"
        
        # Download the file into memory
        response = requests.get(link)
        
        # Untar the file
        tar = tarfile.open(fileobj=response.raw, mode="r:gz")
        tar.extractall()
        tar.close()
