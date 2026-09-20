# 掌握reshape()、squeze()、unsqueeze()、transpose()、permute()、view()、contiguous()等函数使用
import torch
torch.manual_seed(23)

# 1 演示reshape()函数 不改变张量内容
t1 = torch.randint(0, 10, (2, 3))
print(f't1: {t1}, shape: {t1.shape}')
t2 = t1.reshape(3, 2)
print(f't2: {t2}, shape: {t2.shape}') #不改变t1
print('*'*50)
# 2 演示unsqueeze()函数 增加一个维度，升维度
t3 = t1.unsqueeze(0) # 在第0维增加一个维度
t4 = t1.unsqueeze(1) # 在第1维增加一个维度
print(f't3: {t3}, shape: {t3.shape}')
print(f't4: {t4}, shape: {t4.shape}')
print('*'*50)

# 3 演示squeeze()函数 去掉维度为1的维度，降维度 squeeze(d) 只尝试删第 d 维；是 1 就删，不是 1 就原样返回
t6 = torch.randint(0, 10, (1,1,3), dtype=torch.float64)
print(f't6: {t6}, shape: {t6.shape}')
t5 = t6.squeeze() # 去掉维度为1的维度
t7 = t6.squeeze(1) # 去掉第1维的维度
print(f't5: {t5}, shape: {t5.shape}')
print(f't7: {t7}, shape: {t7.shape}')
print('*'*50)
# 4 演示transpose()函数 转置2个维度，permute()函数 转置多个维度
t8 = torch.randint(0, 10, (2, 3, 4))
print(f't8: {t8}, shape: {t8.shape}')
t9 = t8.transpose(0, 1) # 转置第0维和第1维
t10 = t8.permute(2, 0, 1) # 转置第0维和第1维和第2维
print(f't9: {t9}, shape: {t9.shape}')
print(f't10: {t10}, shape: {t10.shape}')
print('*'*50)

# 5 演示view()函数 重新调整张量的形状，要求张量是连续的 contiguous()函数 返回一个连续的张量，若原张量是连续的，则返回原张量，否则返回一个新的连续张量
# is_contiguous()函数 判断张量是否是连续的
t11 = torch.randint(0, 10, (2,3,4))
print(t11.is_contiguous())
print(f't11: {t11}, shape: {t11.shape}')
t12 = t11.view(3,2,4) # 重新调整张量的形状
print(t12.is_contiguous())
print(f't12: {t12}, shape: {t12.shape}')
t13 = t11.transpose(0, 1)
print(f't13: {t13}, shape: {t13.shape}')
print(t13.is_contiguous())
t14 = t13.contiguous().view(2,3,4)
print(f't14: {t14}, shape: {t14.shape}')
