import numpy as np

print("=" * 80)
print("MMFI数据采样方案计算")
print("=" * 80)
print()

# 目标
target_train = 2160
target_test = 540
num_actions = 27

print(f"目标训练集: {target_train}个样本")
print(f"目标测试集: {target_test}个样本")
print(f"动作类别: {num_actions}个")
print()

# 每个动作需要多少样本
samples_per_action_train = target_train // num_actions
samples_per_action_test = target_test // num_actions

print(f"每个动作训练样本: {samples_per_action_train}个")
print(f"每个动作测试样本: {samples_per_action_test}个")
print()

print("=" * 80)
print("方案设计")
print("=" * 80)
print()

# 方案1: 每个动作用固定数量的人
print("方案1: 每个动作用固定数量的人")
print("-" * 80)

# 假设每个人的每个动作有100个样本
samples_per_person = 100

# 计算需要多少人
for train_people in range(1, 20):
    for test_people in range(1, 10):
        # 需要从每个人采样多少个
        if train_people > 0:
            samples_from_each_train = samples_per_action_train / train_people
        else:
            continue

        if test_people > 0:
            samples_from_each_test = samples_per_action_test / test_people
        else:
            continue

        # 检查是否整除且不超过100
        if (samples_from_each_train == int(samples_from_each_train) and
            samples_from_each_test == int(samples_from_each_test) and
            samples_from_each_train <= samples_per_person and
            samples_from_each_test <= samples_per_person):

            total_people = train_people + test_people

            print(f"选项: 训练{train_people}人 + 测试{test_people}人 = 共{total_people}人")
            print(f"  每人训练样本: {int(samples_from_each_train)}个")
            print(f"  每人测试样本: {int(samples_from_each_test)}个")
            print(f"  验证: {train_people}人×{int(samples_from_each_train)}样本×{num_actions}动作 = {train_people * int(samples_from_each_train) * num_actions}训练样本")
            print(f"  验证: {test_people}人×{int(samples_from_each_test)}样本×{num_actions}动作 = {test_people * int(samples_from_each_test) * num_actions}测试样本")
            print()

print()
print("=" * 80)
print("推荐方案")
print("=" * 80)
print()

print("方案A: 8人训练 + 2人测试 (共10人) ⭐⭐⭐⭐⭐")
print("-" * 80)
print("配置:")
print("  - 从40个人中选择10个人")
print("  - 训练人数: S01-S08 (8人)")
print("  - 测试人数: S09-S10 (2人)")
print()
print("采样策略:")
print("  - 每个训练人的每个动作: 取10个样本")
print("  - 每个测试人的每个动作: 取10个样本")
print()
print("数据量:")
print(f"  - 训练集: 8人 × 27动作 × 10样本 = {8*27*10}个样本 ✓")
print(f"  - 测试集: 2人 × 27动作 × 10样本 = {2*27*10}个样本 ✓")
print()
print("优点:")
print("  ✓ 严格按人划分（测试集是完全未见过的2个人）")
print("  ✓ 每个人只用10%的数据（100个样本中取10个）")
print("  ✓ 人数少，计算效率高")
print("  ✓ 测试集能真正评估跨人泛化能力")
print()

print("方案B: 16人训练 + 4人测试 (共20人) ⭐⭐⭐⭐")
print("-" * 80)
print("配置:")
print("  - 从40个人中选择20个人")
print("  - 训练人数: S01-S16 (16人)")
print("  - 测试人数: S17-S20 (4人)")
print()
print("采样策略:")
print("  - 每个训练人的每个动作: 取5个样本")
print("  - 每个测试人的每个动作: 取5个样本")
print()
print("数据量:")
print(f"  - 训练集: 16人 × 27动作 × 5样本 = {16*27*5}个样本 ✓")
print(f"  - 测试集: 4人 × 27动作 × 5样本 = {4*27*5}个样本 ✓")
print()
print("优点:")
print("  ✓ 使用更多人的数据，可能泛化性更好")
print("  ✓ 每个人只用5%的数据")
print()

print("方案C: 32人训练 + 8人测试 (共40人，全用) ⭐⭐⭐")
print("-" * 80)
print("配置:")
print("  - 使用全部40个人")
print("  - 训练人数: S01-S32 (32人)")
print("  - 测试人数: S33-S40 (8人)")
print()
print("采样策略:")
print("  - 每个训练人的每个动作: 取2-3个样本（需要混合）")
print("  - 32人×27动作×2.5样本 = 2160（需要精细控制）")
print()
print("缺点:")
print("  ✗ 采样不够整齐")
print()

print("=" * 80)
print("最终推荐: 方案A (8+2人)")
print("=" * 80)
print()
print("理由:")
print("1. 数据量精确匹配目标")
print("2. 采样策略简单（每人每动作固定10个样本）")
print("3. 严格的按人划分")
print("4. 计算效率高")
print("5. 充分利用每个人的数据多样性（100个样本随机取10个）")
