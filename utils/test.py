# import torch
# print __file__
# print torch.cuda.is_available()
#
# aa = ['c', 'b', 'a']
# bb = [1, 2, 3]
# print aa + bb
# for a, b in zip(aa, bb):
#     print a
#     print b
# ab = zip(aa, bb)
# ab.sort(key=lambda aabb:aabb[1])
# print(ab)
# names = ['001.jpg', '002.jpg', '003.jpg']
# ids = map(lambda name: int(name.split('.')[0]), names)
# print ids
#
# for i in range(10, 5, -1):
#     print i

path = '/home/yjfu/age_exp/AffectNet_index_distill/neutral_index.txt'

index_file = open(path, 'r')
exp_data = []
count = 0
# try:
#     while True:
#         path = index_file.readline()
#         #print path
#         exp_data.append(path)
#         # count += 1
#         # if count % 5000 == 0:
#         #     print count
# except:
#     pass
for line in index_file:
    exp_data.append(line)
print(exp_data.__len__())