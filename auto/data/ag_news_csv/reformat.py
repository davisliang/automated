from os.path import expanduser
import csv
class_dict = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci_Tech"}
file_cnt = 0
with open(expanduser("~/automatedMTL/data/ag_news_csv/train.csv")) as f:
    reader = csv.reader(f)
    for row in reader:
        class_ = row[0]
        content = " ".join(row[1:len(row)])
        with open(expanduser("~/automatedMTL/data/ag_news_csv/Train_raw/"+class_dict[class_]+"/"+str(file_cnt)+".txt"), "w") as f:
            f.write(content)
            f.close()
            file_cnt += 1

print file_cnt
with open(expanduser("~/automatedMTL/data/ag_news_csv/test.csv")) as f:
    reader = csv.reader(f)
    for row in reader:
        class_ = row[0]
        content = " ".join(row[1:len(row)])
        with open(expanduser("~/automatedMTL/data/ag_news_csv/Test_raw/"+class_dict[class_]+"/"+str(file_cnt)+".txt"), "w") as f:
            f.write(content)
            f.close()
            file_cnt += 1
print file_cnt
