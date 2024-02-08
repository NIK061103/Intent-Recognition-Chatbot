import shutil as sh
import pandas as pd

df = pd.read_csv("/Users/nikhilrajput/Downloads/ML DL/deception detection/BagOfLies/Annotations.csv")

df.dropna()


def lambda_eeg(x):
    return "/Users/nikhilrajput/Downloads/ML DL/deception detection/BagOfLies"+x[1:]


df["new_path1"] = df['eeg'].apply(lambda x: lambda_eeg(x))

for j, i in df.iterrows():
    # print(i)
    try:

        if i["truth"] == str(1):
            sh.copyfile(i['new_path1'], f"/Users/nikhilrajput/Downloads/ML DL/deception detection/Truth/eeg{i['usernum']}_{i['run']}")
        else:
            sh.copyfile(i['new_path1'], f"/Users/nikhilrajput/Downloads/ML DL/deception detection/Deception/eeg{i['usernum']}_{i['run']}")
    except Exception as e:
        print("hahah")