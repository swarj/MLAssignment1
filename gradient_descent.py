import pandas as pd
import io
import requests 

url = "https://raw.githubusercontent.com/swarj/MLAssignment1/main/Concrete_Data.xlsx%20-%20Sheet1.csv"             #url where our dataset is being hosted 
download = requests.get(url).content                                                                               #HTTP request to obtain raw data of the .csv file
df = pd.read_csv(io.StringIO(download.decode('utf-8')))                                                            #read .csv into a dataframe 
print (df)
