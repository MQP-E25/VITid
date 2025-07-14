import requests

# URL of the running Flask server
csvURL = "http://127.0.0.1:3000/analyzeCSV"
imgURL = "http://127.0.0.1:3000/analyzeIMG"
nbURL = "http://127.0.0.1:3000/analyzeNotebook"

# Path to the CSV file
csv_path = "test.csv"  
img_path = "test.png"
nb_path  = "testnotebook.csv"

with open(csv_path, "rb") as f:
    csvFiles = {"csv": f}
    response = requests.post(csvURL, files=csvFiles)

print("Status code:", response.status_code)
print("Response for CSV:")
print(response.json())


with open(img_path, "rb") as f:
    imgFiles = {"img": f}
    response = requests.post(imgURL, files=imgFiles)
    
print("Status code:", response.status_code)
print("Response for IMG:")
print(response.json())


with open(nb_path, "rb") as f:
    nbFiles = {"nb": f}
    response = requests.post(nbURL, files=nbFiles)
    
print("Status code:", response.status_code)
print("Response for NB:")
print(response.json())