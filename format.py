import csv

# 读取trains.txt
with open('trains.txt', mode='r', encoding='latin-1') as data_train:
    train = data_train.readlines()
train_deal = []
train_final = [('text', 'label')]

# 格式化
for i in train:
    if i[0:10] == '<Polarity>':
        i = i[10:-12]
        train_deal.append(i)
    if i[0:6] == '<text>':
        i = i[6:-8]
        train_deal.append(i)
for i in range(0, len(train_deal), 2):
    train_final.append((train_deal[i+1], train_deal[i]))

# 存入trains.csv
with open('trains_deal.csv', 'w', encoding='latin-1', newline='') as f:
    csv_writer = csv.writer(f)
    for i in train_final:
        csv_writer.writerow(i)

# 读取tests.txt
with open('tests.txt', mode='r', encoding='latin-1') as data_test:
    test = data_test.readlines()
test_final = [('text', 'label')]

# 格式化
for i in test:
    if i[0:6] == '<text>':
        i = i[6:-8]
        test_final.append((i, -1))

# 存入tests.csv
with open('tests_deal.csv', 'w', encoding='latin-1', newline='') as f:
    csv_writer = csv.writer(f)
    for i in test_final:
        csv_writer.writerow(i)
