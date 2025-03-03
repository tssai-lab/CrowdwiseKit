import math
from collections import defaultdict
import pandas as pd

# 导入自定义工具函数，这些函数用于计算列表的平均值和标准差
from utils import list_mean,list_std,list_mean_2d,list_std_2d
import sys
"""
class OfflineResultsSummary:
    def __init__(self,method):
        # super(ResultsSummary, self).__int__()
        self.method = method
        self.accuracy_results = {}
        self.runtime_results = {}
        self.accuracy_results = defaultdict(lambda :[],self.accuracy_results)
        self.runtime_results = defaultdict(lambda :[], self.runtime_results)
        self.iterations = {}
        self.iterations = defaultdict(lambda :[],self.iterations)
        self.elbos = {}
        self.elbos = defaultdict(lambda :[], self.elbos)

    def add(self,dataset, accuracy, runtime, iteration=1, elbo = None):
        self.accuracy_results[dataset].append(accuracy)
        self.runtime_results[dataset].append(runtime)
        self.iterations[dataset].append(iteration)
        if elbo is not None:
            self.elbos[dataset].append(elbo)


    def get_accuracy_list(self, dataset):
        return self.accuracy_results[dataset]

    def get_runtime_list(self, dataset):
        return self.runtime_results[dataset]

    def get_iteration_list(self, dataset):
        return self.iterations[dataset]

    def get_iteration_mean(self, dataset):
        return list_mean(self.iterations[dataset])

    def get_iteration_std(self, dataset):
        return list_std(self.iterations[dataset])

    def get_num_rounds(self, dataset):
        return len(self.accuracy_results[dataset])

    def get_accuracy_mean(self, dataset):
        return list_mean(self.accuracy_results[dataset])

    def get_accuracy_std(self, dataset):
        return list_std(self.accuracy_results[dataset])

    def get_runtime_mean(self, dataset):
        return list_mean(self.runtime_results[dataset])

    def get_runtime_std(self, dataset):
        return list_std(self.runtime_results[dataset])

    def get_overall_accuracy_mean(self):
        sum_ = 0
        for dataset, list_ in self.accuracy_results.items():
            sum_ += list_mean(list_)
        return sum_/len(self.accuracy_results)

    def get_overall_runtime_mean(self):
        sum_ = 0
        for dataset, list_ in self.runtime_results.items():
            sum_ += list_mean(list_)
        return sum_/len(self.runtime_results)

    def get_method(self):
        return self.method


    def get_EBCC_max_accuracy_runtime_iteration(self,dataset):
        if self.method != 'EBCC':
            raise ValueError('Not EBCC results')
        elbos = self.elbos
        max_val, max_ind = -sys.float_info.max, -sys.float_info.max
        for ind,ele in enumerate(elbos[dataset]):
            if ele > max_val:
                max_val = ele
                max_ind = ind
        return self.accuracy_results[dataset][max_ind], self.runtime_results[dataset][max_ind],self.iterations[dataset][max_ind]

    def get_EBCC_dataframe(self):
        # df = pd.DataFrame()
        # df.columns=['dataset','accuracy','runtime','iteration','elbo']
        if self.method != 'EBCC':
            raise ValueError('Not EBCC results')
        datasets = []
        accuracy_results = []
        runtime_results = []
        iterations = []
        elbos = []
        for dataset in self.accuracy_results.keys():
            for accuracy, runtime, iteration,elbo in zip(self.accuracy_results[dataset],
                                                         self.runtime_results[dataset],
                                                         self.iterations[dataset],
                                                         self.elbos[dataset]):
                # df.append({'accuracy':accuracy, 'runtime':runtime,
                #            'iteration':iteration, 'elbo':elbo},ignore_index=True)
                datasets.append(dataset)
                accuracy_results.append(accuracy)
                runtime_results.append(runtime)
                iterations.append(iteration)
                elbos.append(elbo)
        df = pd.DataFrame({'dataset':datasets,'accuracy':accuracy_results, 'runtime':runtime_results,
                           'iteration':iterations, 'elbo':elbos})
        return df




    def display(self, elbo = False):
        for dataset in self.accuracy_results.keys():
            num_rounds = self.get_num_rounds(dataset)
            acc_mean = round(self.get_accuracy_mean(dataset),4)
            acc_std = round(self.get_accuracy_std(dataset),4)
            runtime_mean = round(self.get_runtime_mean(dataset),4)
            runtime_std = round(self.get_runtime_std(dataset),4)

            iteration_mean = round(self.get_iteration_mean(dataset),4)
            iteration_std = round(self.get_iteration_std(dataset),4)

            message = f'Method {self.method}:Dataset {dataset} runs {num_rounds} rounds, ' \
                      f'accuracy mean {acc_mean}, accuracy std {acc_std},' \
                      f' runtime mean {runtime_mean}, runtime std {runtime_std},' \
                      f' iteration mean {iteration_mean}, iteration std {iteration_std}'
            print(message)

    def to_dataframe_mean(self):
        df = pd.DataFrame({'dataset':list(self.accuracy_results.keys()),
                            'accuracy mean':[round(self.get_accuracy_mean(dataset),4) for dataset in self.accuracy_results.keys()],
                           'accuracy std': [round(self.get_accuracy_std(dataset),4) for dataset in self.accuracy_results.keys()],
                           'runtime mean': [round(self.get_runtime_mean(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                           'runtime std': [round(self.get_runtime_std(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                           'iteration mean': [round(self.get_iteration_mean(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                           'iteration std': [round(self.get_iteration_std(dataset), 4) for dataset in
                                              self.accuracy_results.keys()],
                           })

        return df

    def to_dataframe_detail(self):
        datasets = []
        accuracies = []
        runtimes = []
        iterations = []
        elbos = []
        for dataset in self.accuracy_results.keys():
            for ind, acc in enumerate(self.accuracy_results[dataset]):
                datasets.append(dataset)
                accuracies.append(acc)
                runtimes.append(self.runtime_results[dataset][ind])
                iterations.append(self.iterations[dataset][ind])
                if len(self.elbos[dataset]) > 0:
                    elbos.append(self.elbos[dataset][ind])
        if len(elbos) == 0:
            df = pd.DataFrame({'dataset':datasets,
                               'accuracy':accuracies,
                               'runtime':runtimes,
                               'iteration':iterations})
        else:
            df = pd.DataFrame({'dataset': datasets,
                               'accuracy': accuracies,
                               'runtime': runtimes,
                               'iteration': iterations,
                               'elbo':elbos})
        return df
"""

class OfflineResultsSummary:
    def __init__(self, method):
        # 初始化方法
        self.method = method  # 实验方法的名称
        # 使用 defaultdict 来存储不同数据集的准确率、运行时间、迭代次数和ELBO值
        self.accuracy_results = defaultdict(lambda :[])
        self.runtime_results = defaultdict(lambda :[])
        self.iterations = defaultdict(lambda :[])
        self.elbos = defaultdict(lambda :[])

    def add(self, dataset, accuracy, runtime, iteration=1, elbo=None):
        # 向汇总中添加一个数据集的结果
        self.accuracy_results[dataset].append(accuracy)
        self.runtime_results[dataset].append(runtime)
        self.iterations[dataset].append(iteration)
        if elbo is not None:
            self.elbos[dataset].append(elbo)

    def get_accuracy_list(self, dataset):
        # 获取指定数据集的准确率列表
        return self.accuracy_results[dataset]

    def get_runtime_list(self, dataset):
        # 获取指定数据集的运行时间列表
        return self.runtime_results[dataset]

    def get_iteration_list(self, dataset):
        # 获取指定数据集的迭代次数列表
        return self.iterations[dataset]

    def get_iteration_mean(self, dataset):
        # 计算指定数据集的迭代次数的平均值
        return list_mean(self.iterations[dataset])

    def get_iteration_std(self, dataset):
        # 计算指定数据集的迭代次数的标准差
        return list_std(self.iterations[dataset])

    def get_num_rounds(self, dataset):
        # 获取指定数据集的实验轮数
        return len(self.accuracy_results[dataset])

    def get_accuracy_mean(self, dataset):
        # 计算指定数据集的准确率的平均值
        return list_mean(self.accuracy_results[dataset])

    def get_accuracy_std(self, dataset):
        # 计算指定数据集的准确率的标准差
        return list_std(self.accuracy_results[dataset])

    def get_runtime_mean(self, dataset):
        # 计算指定数据集的运行时间的平均值
        return list_mean(self.runtime_results[dataset])

    def get_runtime_std(self, dataset):
        # 计算指定数据集的运行时间的标准差
        return list_std(self.runtime_results[dataset])

    def get_overall_accuracy_mean(self):
        # 计算所有数据集的准确率的平均值
        sum_ = 0
        for dataset, list_ in self.accuracy_results.items():
            sum_ += list_mean(list_)
        return sum_ / len(self.accuracy_results)

    def get_overall_runtime_mean(self):
        # 计算所有数据集的运行时间的平均值
        sum_ = 0
        for dataset, list_ in self.runtime_results.items():
            sum_ += list_mean(list_)
        return sum_ / len(self.runtime_results)

    def get_method(self):
        # 获取实验方法的名称
        return self.method

    def get_EBCC_max_accuracy_runtime_iteration(self, dataset):
        # 获取EBCC方法中ELBO值最大的准确率、运行时间和迭代次数
        if self.method != 'EBCC':
            raise ValueError('Not EBCC results')
        elbos = self.elbos
        max_val, max_ind = -sys.float_info.max, -sys.float_info.max
        for ind, ele in enumerate(elbos[dataset]):
            if ele > max_val:
                max_val = ele
                max_ind = ind
        return self.accuracy_results[dataset][max_ind], self.runtime_results[dataset][max_ind], self.iterations[dataset][max_ind]

    def get_EBCC_dataframe(self):
        # 创建一个包含EBCC方法结果的DataFrame
        if self.method != 'EBCC':
            raise ValueError('Not EBCC results')
        datasets = []
        accuracy_results = []
        runtime_results = []
        iterations = []
        elbos = []
        for dataset in self.accuracy_results.keys():
            for accuracy, runtime, iteration, elbo in zip(self.accuracy_results[dataset],
                                                         self.runtime_results[dataset],
                                                         self.iterations[dataset],
                                                         self.elbos[dataset]):
                datasets.append(dataset)
                accuracy_results.append(accuracy)
                runtime_results.append(runtime)
                iterations.append(iteration)
                elbos.append(elbo)
        df = pd.DataFrame({'dataset': datasets, 'accuracy': accuracy_results, 'runtime': runtime_results,
                           'iteration': iterations, 'elbo': elbos})
        return df

    def display(self, elbo=False):
        # 打印每个数据集的统计结果
        for dataset in self.accuracy_results.keys():
            num_rounds = self.get_num_rounds(dataset)
            acc_mean = round(self.get_accuracy_mean(dataset), 4)
            acc_std = round(self.get_accuracy_std(dataset), 4)
            runtime_mean = round(self.get_runtime_mean(dataset), 4)
            runtime_std = round(self.get_runtime_std(dataset), 4)
            iteration_mean = round(self.get_iteration_mean(dataset), 4)
            iteration_std = round(self.get_iteration_std(dataset), 4)
            message = f'Method {self.method}: Dataset {dataset} runs {num_rounds} rounds, ' \
                      f'accuracy mean {acc_mean}, accuracy std {acc_std},' \
                      f' runtime mean {runtime_mean}, runtime std {runtime_std},' \
                      f' iteration mean {iteration_mean}, iteration std {iteration_std}'
            print(message)

    def to_dataframe_mean(self):
        # 创建一个包含平均统计结果的DataFrame
        df = pd.DataFrame({'dataset': list(self.accuracy_results.keys()),
                            'accuracy mean': [round(self.get_accuracy_mean(dataset), 4) for dataset in self.accuracy_results.keys()],
                            'accuracy std': [round(self.get_accuracy_std(dataset), 4) for dataset in self.accuracy_results.keys()],
                            'runtime mean': [round(self.get_runtime_mean(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                            'runtime std': [round(self.get_runtime_std(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                            'iteration mean': [round(self.get_iteration_mean(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                            'iteration std': [round(self.get_iteration_std(dataset), 4) for dataset in
                                              self.accuracy_results.keys()],
                            })
        return df

    def to_dataframe_detail(self):
        # 创建一个包含详细结果的DataFrame
        datasets = []
        accuracies = []
        runtimes = []
        iterations = []
        elbos = []
        for dataset in self.accuracy_results.keys():
            for ind, acc in enumerate(self.accuracy_results[dataset]):
                datasets.append(dataset)
                accuracies.append(acc)
                runtimes.append(self.runtime_results[dataset][ind])
                iterations.append(self.iterations[dataset][ind])
                if len(self.elbos[dataset]) > 0:
                    elbos.append(self.elbos[dataset][ind])
        if len(elbos) == 0:
            df = pd.DataFrame({'dataset': datasets,
                               'accuracy': accuracies,
                               'runtime': runtimes,
                               'iteration': iterations})
        else:
            df = pd.DataFrame({'dataset': datasets,
                               'accuracy': accuracies,
                               'runtime': runtimes,
                               'iteration': iterations,
                               'elbo': elbos})
        return df

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
"""
class OnlineResultsSummary:
    def __init__(self,method, num_chunks = 10):
        # super(ResultsSummary, self).__int__()
        self.method = method
        self.num_chunks = num_chunks
        self.accuracy_results = {}
        self.runtime_results = {}
        self.accuracy_results = defaultdict(lambda :[],self.accuracy_results)
        self.runtime_results = defaultdict(lambda :[], self.runtime_results)
        # self.iterations = {}
        # self.iterations = defaultdict(lambda :[],self.iterations)

    def add(self,dataset, accuracy_list, runtime_list):
        self.accuracy_results[dataset].append(accuracy_list)
        self.runtime_results[dataset].append(runtime_list)
        # self.iterations[dataset].append(iteration)

    def get_accuracy_list(self, dataset):
        return self.accuracy_results[dataset]

    def get_runtime_list(self, dataset):
        return self.runtime_results[dataset]

    # def get_iteration_list(self, dataset):
    #     return self.iterations[dataset]

    # def get_iteration_mean(self, dataset):
    #     return list_mean(self.iterations[dataset])
    #
    # def get_iteration_std(self, dataset):
    #     return list_std(self.iterations[dataset])

    def get_num_rounds(self, dataset):
        return len(self.accuracy_results[dataset])

    def get_accuracy_mean(self, dataset):
        # return list_mean(self.accuracy_results[dataset])
        return list_mean_2d(self.accuracy_results[dataset],axis=0)

    def get_accuracy_std(self, dataset):
        # return list_std(self.accuracy_results[dataset])
        return list_std_2d(self.accuracy_results[dataset], axis=0)

    def get_runtime_mean(self, dataset):
        # return list_mean(self.runtime_results[dataset])
        return list_mean_2d(self.runtime_results[dataset], axis=0)

    def get_runtime_std(self, dataset):
        # return list_std(self.runtime_results[dataset])
        return list_std_2d(self.runtime_results[dataset], axis=0)



    def get_method(self):
        return self.method

    def to_dataframe_accuracy_all_mean(self):
        entries = []
        for dataset in self.accuracy_results.keys():
            datset_accuracy_mean = self.get_accuracy_mean(dataset)
            entries.append(datset_accuracy_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.accuracy_results.keys()))
        return df

    def to_dataframe_runtime_all_mean(self):
        entries = []
        for dataset in self.runtime_results.keys():
            datset_runtime_mean = self.get_runtime_mean(dataset)
            entries.append(datset_runtime_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.runtime_results.keys()))
        return df

    def to_dataframe_accuracy_all_std(self):
        entries = []
        for dataset in self.accuracy_results.keys():
            datset_accuracy_mean = self.get_accuracy_std(dataset)
            entries.append(datset_accuracy_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.accuracy_results.keys()))
        return df

    def to_dataframe_runtime_all_std(self):
        entries = []
        for dataset in self.runtime_results.keys():
            datset_runtime_mean = self.get_runtime_std(dataset)
            entries.append(datset_runtime_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.runtime_results.keys()))
        return df

    def to_dataframe_accuracy_detail(self):

        accuracy_dict = self.accuracy_results
        num_chunks = 10
        entries = []
        for dataset in accuracy_dict.keys():
            for ind, acc in enumerate(accuracy_dict[dataset]):
                row = [dataset]
                row += acc
                entries.append(row)
        columns = ['dataset']
        for i in range(num_chunks):
            columns.append(str(i + 1))
        df = pd.DataFrame(data=entries, columns=columns)
        return df

    def to_dataframe_runtime_detail(self):

        runtime_dict = self.runtime_results
        num_chunks = 10
        entries = []
        for dataset in runtime_dict.keys():
            for ind, acc in enumerate(runtime_dict[dataset]):
                row = [dataset]
                row += acc
                entries.append(row)
        columns = ['dataset']
        for i in range(num_chunks):
            columns.append(str(i + 1))
        df = pd.DataFrame(data=entries, columns=columns)
        return df
"""

class OnlineResultsSummary:
    def __init__(self, method, num_chunks=10):
        """
        初始化函数。
        :param method: 字符串，表示测试的方法名称。
        :param num_chunks: 整数，默认值为10，表示将结果分成多少块（可能用于多次运行或迭代）。
        """
        self.method = method
        self.num_chunks = num_chunks
        # 初始化准确率和运行时间的结果字典，但下面的代码有误，重复赋值且使用了错误的初始化方式
        # 正确的方式是单独使用defaultdict，不需要先赋值为空字典
        self.accuracy_results = defaultdict(lambda: [])  # 为每个数据集存储一个准确率的列表列表
        self.runtime_results = defaultdict(lambda: [])  # 为每个数据集存储一个运行时间的列表列表
        # 下面的代码是重复的，并且错误地尝试修改已经初始化的defaultdict
        # self.accuracy_results = defaultdict(lambda :[],self.accuracy_results)
        # self.runtime_results = defaultdict(lambda :[], self.runtime_results)

    def add(self, dataset, accuracy_list, runtime_list):
        """
        添加数据集的结果。
        :param dataset: 字符串，表示数据集的名称。
        :param accuracy_list: 列表，包含当前数据集的准确率结果（可能是多次迭代的准确率）。
        :param runtime_list: 列表，包含当前数据集的运行时间结果（可能是多次迭代的运行时间）。
        """
        self.accuracy_results[dataset].append(accuracy_list)
        self.runtime_results[dataset].append(runtime_list)

    def get_accuracy_list(self, dataset):
        """
        获取指定数据集的准确率结果列表。
        :param dataset: 字符串，表示数据集的名称。
        :return: 列表的列表，包含所有记录的准确率结果。
        """
        return self.accuracy_results[dataset]

    def get_runtime_list(self, dataset):
        """
        获取指定数据集的运行时间结果列表。
        :param dataset: 字符串，表示数据集的名称。
        :return: 列表的列表，包含所有记录的运行时间结果。
        """
        return self.runtime_results[dataset]

    def get_num_rounds(self, dataset):
        """
        获取指定数据集的结果记录次数（可能是迭代次数）。
        :param dataset: 字符串，表示数据集的名称。
        :return: 整数，表示结果记录的次数。
        """
        return len(self.accuracy_results[dataset])

    def get_accuracy_mean(self, dataset):
        """
        计算指定数据集准确率的平均值（假设准确率结果是二维列表，每内层列表表示一次迭代的准确率）。
        :param dataset: 字符串，表示数据集的名称。
        :return: 列表，包含每个块的准确率平均值。
        """
        return list_mean_2d(self.accuracy_results[dataset], axis=0)

    def get_accuracy_std(self, dataset):
        """
        计算指定数据集准确率的标准差（假设准确率结果是二维列表，每内层列表表示一次迭代的准确率）。
        :param dataset: 字符串，表示数据集的名称。
        :return: 列表，包含每个块的准确率标准差。
        """
        return list_std_2d(self.accuracy_results[dataset], axis=0)

    def get_runtime_mean(self, dataset):
        """
        计算指定数据集运行时间的平均值（假设运行时间结果是二维列表，每内层列表表示一次迭代的运行时间）。
        :param dataset: 字符串，表示数据集的名称。
        :return: 列表，包含每个块的运行时间平均值。
        """
        return list_mean_2d(self.runtime_results[dataset], axis=0)

    def get_runtime_std(self, dataset):
        """
        计算指定数据集运行时间的标准差（假设运行时间结果是二维列表，每内层列表表示一次迭代的运行时间）。
        :param dataset: 字符串，表示数据集的名称。
        :return: 列表，包含每个块的运行时间标准差。
        """
        return list_std_2d(self.runtime_results[dataset], axis=0)

    def get_method(self):
        """
        获取测试的方法名称。
        :return: 字符串，表示测试的方法名称。
        """
        return self.method

    def to_dataframe_accuracy_all_mean(self):
        """
        将所有数据集的准确率平均值转换为pandas DataFrame。
        :return: DataFrame，包含每个数据集的准确率平均值。
        """
        entries = []
        for dataset in self.accuracy_results.keys():
            datset_accuracy_mean = self.get_accuracy_mean(dataset)
            entries.append(datset_accuracy_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.accuracy_results.keys()))
        return df

    def to_dataframe_runtime_all_mean(self):
        """
        将所有数据集的运行时间平均值转换为pandas DataFrame。
        :return: DataFrame，包含每个数据集的运行时间平均值。
        """
        entries = []
        for dataset in self.runtime_results.keys():
            datset_runtime_mean = self.get_runtime_mean(dataset)
            entries.append(datset_runtime_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.runtime_results.keys()))
        return df

    def to_dataframe_accuracy_all_std(self):
        """
        将所有数据集的准确率标准差转换为pandas DataFrame。
        :return: DataFrame，包含每个数据集的准确率标准差。
        """
        entries = []
        for dataset in self.accuracy_results.keys():
            datset_accuracy_mean = self.get_accuracy_std(dataset)
            entries.append(datset_accuracy_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.accuracy_results.keys()))
        return df

    def to_dataframe_runtime_all_std(self):
        """
        将所有数据集的运行时间标准差转换为pandas DataFrame。
        :return: DataFrame，包含每个数据集的运行时间标准差。
        """
        entries = []
        for dataset in self.runtime_results.keys():
            datset_runtime_mean = self.get_runtime_std(dataset)
            entries.append(datset_runtime_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.runtime_results.keys()))
        return df

    def to_dataframe_accuracy_detail(self):
        """
        将所有数据集的详细准确率结果转换为pandas DataFrame。
        :return: DataFrame，包含每个数据集每次迭代的准确率。
        """
        accuracy_dict = self.accuracy_results
        num_chunks = 10  # 注意：这里硬编码了num_chunks的值，但实际上它应该与实例变量self.num_chunks一致
        entries = []
        for dataset in accuracy_dict.keys():
            for ind, acc in enumerate(accuracy_dict[dataset]):
                row = [dataset]
                row += acc
                entries.append(row)
        columns = ['dataset']
        for i in range(num_chunks):
            columns.append(str(i + 1))
        df = pd.DataFrame(data=entries, columns=columns)
        return df

    def to_dataframe_runtime_detail(self):
        """
        将所有数据集的详细运行时间结果转换为pandas DataFrame。
        :return: DataFrame，包含每个数据集每次迭代的运行时间。
        """
        runtime_dict = self.runtime_results
        num_chunks = 10  # 同样，这里硬编码了num_chunks的值
        entries = []
        for dataset in runtime_dict.keys():
            for ind, acc in enumerate(runtime_dict[dataset]):
                row = [dataset]
                row += acc
                entries.append(row)
        columns = ['dataset']
        for i in range(num_chunks):
            columns.append(str(i + 1))
        df = pd.DataFrame(data=entries, columns=columns)
        return df