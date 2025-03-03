import csv  # 导入csv模块，用于处理CSV文件
import random  # 导入random模块，用于生成随机数或进行随机操作


# 定义一个函数，用于将列表分割成n_parts个部分
def list_split(list_, n_parts):
    k, m = divmod(len(list_), n_parts)  # 使用divmod函数计算每个部分的基本长度k和剩余元素数m
    # 使用列表推导式，根据计算出的k和m，将原列表分割成n_parts个部分
    return [list_[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]


# 定义一个函数，用于生成数据块（chunks）
def chunks_generation(label_path, truth_path, num_chunks=10):
    # 调用gete2wlandw2el函数，获取没有ground truth的items（e2wl），以及其他两个未使用的返回值
    e2wl, _, _ = gete2wlandw2el(label_path)
    items_without_ground_truths = list(e2wl.keys())  # 获取没有ground truth的items列表
    items_count = len(items_without_ground_truths)  # 计算没有ground truth的items数量

    # 初始化有ground truth的items列表和ground truths字典
    items_with_ground_truths = []
    ground_truths = {}
    # 打开ground truth文件，读取内容
    with open(truth_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过文件的第一行（通常是标题行）
        for line in reader:
            item, truth = line  # 读取每一行，得到item和对应的truth
            ground_truths[item] = truth  # 将item和truth存入字典
            if int(truth) >= 0:  # 如果truth是非负数
                items_with_ground_truths.append(item)  # 将item添加到有ground truth的列表中
                items_without_ground_truths.remove(item)  # 从没有ground truth的列表中移除该item

    # 随机打乱没有和有ground truth的items列表
    random.shuffle(items_without_ground_truths)
    random.shuffle(items_with_ground_truths)

    # 将两个列表分别分割成num_chunks个部分
    items_without_ground_truths_splited = list_split(items_without_ground_truths, n_parts=num_chunks)
    items_with_ground_truths_splited = list_split(items_with_ground_truths, n_parts=num_chunks)

    chunks = []  # 初始化chunks列表，用于存储最终的数据块
    for i in range(num_chunks):
        # 将有和无ground truth的items合并，然后随机打乱，存入chunks列表
        chunk = items_with_ground_truths_splited[i] + items_without_ground_truths_splited[i]
        random.shuffle(chunk)
        chunks.append(chunk)
    return chunks  # 返回生成的chunks


# 定义一个函数，用于从CSV文件中获取e2wl（example to worker and label的映射），w2el（worker to example and label的映射），和label集合
def gete2wlandw2el(datafile):
    e2wl = {}  # 初始化e2wl字典
    w2el = {}  # 初始化w2el字典
    label_set = []  # 初始化label集合

    with open(datafile, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过文件的第一行（通常是标题行）
        for line in reader:
            example, worker, label = line  # 读取每一行，得到example, worker, label
            if example not in e2wl:
                e2wl[example] = []  # 如果example不在e2wl中，为其初始化一个空列表
            e2wl[example].append([worker, label])  # 将worker和label添加到example对应的列表中

            if worker not in w2el:
                w2el[worker] = []  # 如果worker不在w2el中，为其初始化一个空列表
            w2el[worker].append([example, label])  # 将example和label添加到worker对应的列表中

            if label not in label_set:
                label_set.append(label)  # 如果label不在label集合中，将其添加到集合中

    return e2wl, w2el, label_set  # 返回e2wl, w2el, label_set


# 定义一个函数，用于计算准确率
def getaccuracy(truthfile, truths, chunk=None):
    ground_truths = {}  # 初始化ground_truths字典
    with open(truthfile, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过文件的第一行（通常是标题行）
        for line in reader:
            item, truth = line  # 读取每一行，得到item和对应的truth
            ground_truths[item] = truth  # 将item和truth存入字典

    ccount, tcount = 0, 0  # 初始化正确计数和总计数
    for item, ground_truth in ground_truths.items():
        if chunk is not None:
            # 如果指定了chunk，并且当前item不在chunk中，则跳过
            if item not in chunk:
                continue
        if int(ground_truth) < 0:
            continue  # 如果ground truth是负数，则跳过
        if truths.get(item) is None:
            continue  # 如果当前item的预测值不存在，则跳过
        aggregated_truth = truths[item]  # 获取当前item的预测值
        tcount += 1  # 总计数加1
        if aggregated_truth == ground_truth:  # 如果预测值等于ground truth
            ccount += 1  # 正确计数加1
    return ccount * 1.0 / tcount  # 返回准确率


if __name__ == '__main__':
    # 测试gete2wlandw2el函数
    e2wl, w2el, label_set = gete2wlandw2el("./data/crowdscale2013/fact_eval/label.csv")