import numpy as np

N = 10000  # 模拟次数

# 随机放置汽车（0,1,2 号门）
car = np.random.randint(0, 3, N)

# 玩家随机选一扇门
choice = np.random.randint(0, 3, N)

# 不换门：选中汽车就赢
stay_win = (choice == car).sum()

# 换门：没选中汽车就赢（因为另一扇山羊被主持人开了，换过去一定是车）
switch_win = (choice != car).sum()

print(f"模拟 {N} 次三门问题")
print(f"不换门: 赢 {stay_win} 次, 概率 = {stay_win / N:.3f}")
print(f"换  门: 赢 {switch_win} 次, 概率 = {switch_win / N:.3f}")
print(f"\n换门胜率是不换的 {switch_win / stay_win:.1f} 倍, 理论值 2.0 倍")
