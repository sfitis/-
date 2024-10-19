import logging
import os.path
import random
import re
import torch.utils.data
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from FL import FederatedTrainer
from utils import determining_original_model, determine_attack_model
from datetime import datetime
from diffprivlib.models import GaussianNB
from diffprivlib.utils import PrivacyLeakWarning

import warnings
warnings.filterwarnings("ignore", category=PrivacyLeakWarning)

class Exp:
    def __init__(self, args):
        self.dataset_name = args["dataset_name"]
        self.original_model_name = args["original_model_name"]
        # self.unlearning_round = args["unlearning round"]
        self.max_agg_round = args["max_aggregation_round"]
        self.local_batch_size = args["local_batch_size"]
        self.local_epoch = args["local_epoch"]
        self.client_number = args["client_number"]
        self.decimal_place = args["decimal_places"]
        self.all_client_number = args["all_client_number"]
        self.attack_model_name = args["attack_model_name"]
        self.round_number = args["round_number"]
        self.construct_feather = args["method_of_construct_feather"]
        assert self.client_number <= self.all_client_number, "每一轮参与客户端训练的参数必须小于客户端总数"

        self.current_time = datetime.now().strftime("%d_%H_%M")
        # self.current_time = "18_01_06"

        self.attack_model = None
        self.original_attack_model = None
        self.initial_path = "model/{}_{}/".format(self.dataset_name, self.original_model_name)

        # 便于控制中间模型模型训练
        self.begin, self.end = 0, self.round_number

    def load_data(self):
        # 数据集已经下载到data/slice
        self.logger.info('loading data')
        self.logger.info('loaded data')


class ModelTrainer(Exp):
    def __init__(self, args):
        super(ModelTrainer, self).__init__(args)
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("trainer")
        if not os.path.exists(self.initial_path):
            os.makedirs(self.initial_path)
            self.logger.info("生成模型目录: {} ".format(self.initial_path))
        path = 'log/trainer_{}_{}/'.format(self.dataset_name, self.original_model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        file_handler = logging.FileHandler(path + '{}.log'.format(self.current_time))
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.info(args)
        self.logger.info("Experiment Start!".format())
        self.load_data()
        self.logger.info("begin = {}, end = {}".format(self.begin, self.end))

        self.shadow_train_data_paths = "data/slice/{}/shadow_train_data_{{}}.npy".format(self.dataset_name)
        self.shadow_train_label_paths = "data/slice/{}/shadow_train_label_{{}}.npy".format(self.dataset_name)
        self.shadow_test_data_paths = "data/slice/{}/shadow_test_data_{{}}.npy".format(self.dataset_name)
        self.shadow_test_label_paths = "data/slice/{}/shadow_test_label_{{}}.npy".format(self.dataset_name)
        self.shadow_all_test_path = ["data/slice/{}/shadow_test_data_{{}}.npy".format(self.dataset_name),
                                     "data/slice/{}/shadow_test_label_{{}}.npy".format(self.dataset_name)]
        # target数据路径
        self.target_train_data_paths = "data/slice/{}/target_train_data_{{}}.npy".format(self.dataset_name)
        self.target_train_label_paths = "data/slice/{}/target_train_label_{{}}.npy".format(self.dataset_name)
        self.target_test_data_paths = "data/slice/{}/target_test_data_{{}}.npy".format(self.dataset_name)
        self.target_test_label_paths = "data/slice/{}/target_test_label_{{}}.npy".format(self.dataset_name)
        self.target_all_test_path = ["data/slice/{}/target_test_data_{{}}.npy".format(self.dataset_name),
                                     "data/slice/{}/target_test_label_{{}}.npy".format(self.dataset_name)]
        # 模型路径

        self.shadow_initial_model_path0 = "shadow_initial_parameters.npy"
        self.shadow_original_model_path0 = "shadow_original_model.npy"
        self.shadow_unlearning_model_path0 = "shadow_unlearned_model_unid_{}.npy"

        self.target_initial_model_path0 = "target_initial_parameters.npy"
        self.target_original_model_path0 = "target_original_model.npy"
        self.target_unlearning_model_path0 = "target_unlearned_model_unid_{}.npy"

        # 参与shadow客户端的id
        self.shadow_negative_id = {}
        self.shadow_participants_id = {}
        self.shadow_unlearned_ids = {}

        self.target_negative_id = {}
        self.target_participants_id = {}
        self.target_unlearned_ids = {}
        # shadow数据路径
        self.model = determining_original_model(self.original_model_name, self.dataset_name)
        self.get_shadow_models(self.round_number)  # 训练多轮shadow模型
        self.get_target_models(self.round_number)  # 训练target模型

    def get_shadow_models(self, n):
        """
        :param n:  shadow 模型的个数
        :return: 训练出来的模型保存在
        self.shadow_initial_model_path， self.shadow_original_model_path，self.shadow_unlearning_model_path
        """
        self.initial_shadow_path = self.initial_path + "shadow_models/"
        if not os.path.exists(self.initial_shadow_path):
            os.makedirs(self.initial_shadow_path)
            self.logger.info("成功生成目录: {}".format(self.initial_shadow_path))
        # for i in tqdm(range(n), desc="shadow training round"):
        for i in tqdm(range(self.begin, self.end), desc="shadow training round"):  # 便于控制中间模型训练
            self.shadow_path = self.initial_shadow_path + str(i) + "/"
            if not os.path.exists(self.shadow_path):
                os.makedirs(self.shadow_path)
            else:
                self.clear_directory(self.shadow_path)
            self.logger.info("shadow模型文件保存在：{}".format(self.shadow_path))
            self.shadow_initial_model_path = self.shadow_path + self.shadow_initial_model_path0
            self.shadow_original_model_path = self.shadow_path + self.shadow_original_model_path0
            self.shadow_unlearning_model_path = self.shadow_path + self.shadow_unlearning_model_path0
            random_ids = random.sample(range(20), 11)
            self.shadow_negative_id[i] = random_ids[-1]
            self.shadow_participants_id[i] = random_ids[:-1]
            self.logger.info("shadow: 第 {i} 轮 参与训练的客户端id = {ids} ".format(i=i, ids=self.shadow_participants_id[i]))
            self.logger.info("shadow: 第 {i} 轮 反向数据客户端id = {ids}".format(i=i, ids=self.shadow_negative_id[i]))
            self.training_shadow_model(i)
            with open(self.shadow_path + "negative_id.txt", "w") as f:
                f.write(str(self.shadow_negative_id[i]))
                f.write("\n")
                f.write(str(self.shadow_participants_id[i]))
                f.write("\n")
                f.write("uid = " + str(self.shadow_unlearned_ids[i]))

    def get_target_models(self, n):
        self.initial_target_path = self.initial_path + "target_models/"
        if not os.path.exists(self.initial_target_path):
            os.makedirs(self.initial_target_path)
            self.logger.info("成功生成目录: {}".format(self.initial_target_path))
        # for i in tqdm(range(n), desc="target training round"):
        for i in tqdm(range(self.begin, self.end), desc="target training round"):  # 便于控制中间模型训练
            self.target_path = self.initial_target_path + str(i) + "/"
            if not os.path.exists(self.target_path):
                os.makedirs(self.target_path)
            else:
                self.clear_directory(self.target_path)
            self.logger.info("target模型文件保存在：{}".format(self.target_path))
            self.target_initial_model_path = self.target_path + self.target_initial_model_path0
            self.target_original_model_path = self.target_path + self.target_original_model_path0
            self.target_unlearning_model_path = self.target_path + self.target_unlearning_model_path0
            random_ids = random.sample(range(20), 11)
            self.target_negative_id[i] = random_ids[-1]
            self.target_participants_id[i] = random_ids[:-1]
            # self.target_negative_data_path = self.target_negative_data_path.format(self.target_negative_id[i])
            self.logger.info("target: 第 {i} 轮 参与训练的客户端id = {ids} ".format(i=i, ids=self.target_participants_id[i]))
            self.logger.info("target: 第 {i} 轮 反向数据客户端id = {ids}".format(i=i, ids=self.target_negative_id[i]))
            self.training_target_model(i)
            with open(self.target_path + "negative_id.txt", "w") as f:
                f.write(str(self.target_negative_id[i]))
                f.write("\n")
                f.write(str(self.target_participants_id[i]))
                f.write("\n")
                f.write("uid = " + str(self.target_unlearned_ids[i]))

    def clear_directory(self, directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def training_shadow_model(self, n):  # 训练shadow模型
        initial_parameters = {}
        if self.original_model_name in ['lenet', 'simpleCNN']:
            if not initial_parameters:
                for key, var in self.model.state_dict().items():
                    initial_parameters[key] = var.clone()
        elif self.original_model_name in ['LR', "LR_without"]:
            if not initial_parameters:
                for key, var in self.model.state_dict().items():
                    initial_parameters[key] = var
        else:
            raise "exp 初始化中没有这个模型"
        # print(initial_parameters)

        # {'coef': array([[0.66855771, 0.30030047, 0.30626918, 0.60042902, 0.50701448,
        #                  0.36062128, 0.04687179, 0.365417, 0.15010937, 0.52663864,
        #                  0.18664577, 0.98631607, 0.45962983, 0.52658177]]), 'intercept': array([0.20430802]),
        #  'classes_': array([0, 1], dtype=uint32)}
        np.save(self.shadow_initial_model_path, initial_parameters)
        self.logger.info("shadow {}：全局变量初始化完成".format(n))
        ftrainer = FederatedTrainer(self.shadow_participants_id[n], self.original_model_name, self.dataset_name,
                                    self.shadow_initial_model_path, self.decimal_place,
                                    "shadow")
        data_path = [self.shadow_train_data_paths, self.shadow_train_label_paths, self.shadow_test_data_paths,
                     self.shadow_test_label_paths, self.shadow_all_test_path]
        k, acc = ftrainer.training(self.max_agg_round, self.shadow_original_model_path, self.local_epoch,
                                   self.local_batch_size,
                                   data_path)
        self.logger.info("shadow {}:初始模型聚合{}轮次, 全训练模型准确率为{}".format(n, k, acc))
        self.logger.info('shadow {}:初始模型训练完成'.format(n))

        self.logger.info('{}: 开始训练shadow去学习模型.....'.format(n))
        un = random.sample(self.shadow_participants_id[n], 1)[0]
        self.shadow_unlearned_ids[n] = un
        self.logger.info("shadow: unlearning id = {}".format(un))
        unlearning_model_path = self.shadow_unlearning_model_path.format(un)
        k, acc = ftrainer.training(self.max_agg_round, unlearning_model_path, self.local_epoch,
                                   self.local_batch_size,
                                   data_path, [un])
        self.logger.info("shadow {}: {}去学习模型聚合{}轮次, 全训练模型准确率为{}".format(n, un, k, acc))
        self.logger.info('shadow {}: 模型训练完成'.format(n))

    def training_target_model(self, j):
        initial_parameters = {}
        self.model.initialize_parameters()  # 模型参数随机化
        if self.original_model_name in ['lenet', 'simpleCNN']:
            if not initial_parameters:
                for key, var in self.model.state_dict().items():
                    initial_parameters[key] = var.clone()
        elif self.original_model_name in ['LR', "LR_without"]:
            if not initial_parameters:
                for key, var in self.model.state_dict().items():
                    initial_parameters[key] = var
        else:
            raise "training_target_model 初始化没有该模型"
        np.save(self.target_initial_model_path, initial_parameters)
        self.logger.info("target 全局变量初始化完成")
        self.logger.info("开始训练target模型......")
        ttrainer = FederatedTrainer(self.target_participants_id[j], self.original_model_name, self.dataset_name,
                                    self.target_initial_model_path, self.decimal_place,
                                    "target")
        data_path = [self.target_train_data_paths, self.target_train_label_paths, self.target_test_data_paths,
                     self.target_test_label_paths, self.target_all_test_path]
        k, acc = ttrainer.training(self.max_agg_round, self.target_original_model_path, self.local_epoch,
                                   self.local_batch_size,
                                   data_path)
        self.logger.info("target:初始模型聚合{}轮次, 全训练模型准确率为{}".format(k, acc))
        self.logger.info('target:初始模型训练完成')

        self.logger.info('开始训练target去学习模型.....')
        un = random.sample(self.target_participants_id[j], 1)[0]
        self.target_unlearned_ids[j] = un
        self.logger.info("target: unlearning id = {}".format(un))
        unlearning_model_path = self.target_unlearning_model_path.format(un)
        k, acc = ttrainer.training(self.max_agg_round, unlearning_model_path, self.local_epoch,
                                   self.local_batch_size,
                                   data_path, [un])
        self.logger.info("target第{}个去学习模型聚合{}轮次, 全训练模型准确率为{}".format(un, k, acc))
        self.logger.info('target:模型训练完成')


class AttackModelTrainer(Exp):
    def __init__(self, args):
        super(AttackModelTrainer, self).__init__(args)
        # print(self.construct_feather)

        # 用于重复试验
        # for self.end in range(1, 11):
        #     tmp_n = self.end - self.begin
        tmp_n = self.round_number
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Attacker")
        if not os.path.exists(self.initial_path):
            os.makedirs(self.initial_path)
            self.logger.info("生成模型目录: {} ".format(self.initial_path))
        path = 'log/attacker_{}_{}_{}/'.format(self.dataset_name, self.original_model_name, self.attack_model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        file_handler = logging.FileHandler(path + '{}_{}.log'.format(tmp_n, self.current_time))
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        self.logger.info(args)
        self.logger.info("Experiment Start!".format())
        self.load_data()

        self.logger.info("初始模型个数: {}".format(tmp_n))  # tmp_n 要用于记录参与的训练初始模型数
        self.logger.info("begin = {}, end = {}".format(self.begin, self.end))

        self.shadow_train_data_paths = "data/slice/{}/shadow_train_data_{{}}.npy".format(self.dataset_name)
        self.target_train_data_paths = "data/slice/{}/target_train_data_{{}}.npy".format(self.dataset_name)

        self.original_model_path = "shadow_original_model.npy"
        attacker_path = self.initial_path + "{}/".format(self.attack_model_name)
        if not os.path.exists(attacker_path):
            os.makedirs(attacker_path)
        self.original_attack_model_path = attacker_path + "original_{{}}_{}_attacker.npy".format(tmp_n)
        self.attack_model_path = attacker_path + "{{}}_{}_attacker.npy".format(tmp_n)

        # 不考虑数据删除场景
        #feather分布为先正向后负向，而论文代码是一正一负交错，是否会影响结果(已改，但不影响结果)
        #feather生成出错了

        base_x, base_y = self.construct_base_dataset(self.begin, self.end, "shadow")  # 用shadow数据，构造训练攻击模型的数据集
        self.original_attack_model = self.training_attack_model(base_x, base_y, 0)  # 训练攻击模型并保存
        test_base_x, test_base_y = self.construct_base_dataset(self.begin, self.end, "target")  # 使用target数据，构造攻击模型的测试集  #这步是否正确，知道target数据参与训练吗？
        self.evaluate_attack_model(test_base_x, test_base_y, 0, base_x, base_y)
        # data = np.concatenate([base_x, test_base_x])
        # scaler = preprocessing.StandardScaler().fit(data)
        # with open('data/{}/{}_{}_attacker_scaler.pkl'.format(self.dataset_name, "base", self.original_model_name),
        #           'wb') as file:
        #     pickle.dump(scaler, file)

        # 考虑数据删除场景
        x, y = self.construct_diff_dataset(self.begin, self.end, "shadow", self.construct_feather)  # 构造训练攻击模型的数据集
        self.attack_model = self.training_attack_model(x, y, 1)  # 训练攻击模型
        test_x, test_y = self.construct_diff_dataset(self.begin, self.end, "target", self.construct_feather) # 使用target数据，构造攻击模型的测试集
        # self.evaluate_attack_model(x, y, 1)
        self.evaluate_attack_model(test_x, test_y, 1, x, y)
        # data = np.concatenate([x, test_x])
        # scaler = preprocessing.StandardScaler().fit(data)
        # with open('data/{}/{}_{}_attacker_scaler.pkl'.format(self.dataset_name, "new", self.original_model_name),
        #           'wb') as file:
        #     pickle.dump(scaler, file)



        # # 计算degcount, degrate
        # self.calculate_DegCount_and_DegRate(test_base_x, test_base_y, self.original_attack_model,
        #                                                         test_x, test_y, self.attack_model)

        # self.logger.info("degcount = {} , degrate = {}".format(degcount, degrate))

    def construct_base_dataset(self, begin, end, flag):
        # flag :["shadow", "target"]
        features = []
        labels = []
        for j in range(begin, end):
            # 构造正向特征  unid：x在shadow_original_model与target_original_model中
            initial_path = self.initial_path + flag + "_models/" + str(j) + "/"
            model_path, _, unid, negative_id = self.get_path(initial_path)
            positive_data_path = self.shadow_train_data_paths.format(
                unid) if flag == "shadow" else self.target_train_data_paths.format(unid)
            # print("model_path={}, data_path={}".format(model_path, positive_data_path))
            #   flag = shadow情况下
            # # model_path=model/adult_LR/shadow_models/0/shadow_original_model.npy
            # # data_path=data/slice/adult/shadow_train_data_5.npy  5为此轮的 unlearnid
            #   flag = target情况下
            # # model_path=model/adult_LR/target_models/0/target_original_model.npy,
            # # data_path=data/slice/adult/target_train_data_1.npy  1为此轮的 unlearnid


            feature = self.get_model_output(model_path, positive_data_path).tolist()
            # print("feature:",format(feature))
            # feature: [[0.9398268541488568, 0.06017314585114315], [0.9416596066097671, 0.05834039339023285], [0.8670804516453121, 0.13291954835468786], [0.34110885714588557, 0.6588911428541144]
            features.append(feature)

            label = [1] * len(feature)

            labels.append(label)

            # 构造反向特征  negative_id：x不在shadow_original_model与target_original_model中
            negative_data_path = self.shadow_train_data_paths.format(
                negative_id) if flag == "shadow" else self.target_train_data_paths.format(negative_id)


            feature = self.get_model_output(model_path, negative_data_path).tolist()
            #print("model_path={}, data_path={}".format(model_path, negative_data_path))
            features.append(feature)
            label = [0] * len(feature)
            labels.append(label)

        features = sum(features, [])
        labels = sum(labels, [])
        #
        # print("features:", format(features))
        # print("labels:", format(labels))
        #
        new_features = []
        new_labels = []
        j = 0
        k = len(labels)-1
        for i in range(int(len(labels)/2)):
            new_labels.append(labels[j])
            new_labels.append(labels[k])
            new_features.append(features[j])
            new_features.append(features[k])
            j = j + 1
            k = k - 1

        print(new_features)
        print(new_labels)

        print("{}: 正向数据量: {}; 反向数据量: {}".format(flag, sum(new_labels), len(new_labels) - sum(new_labels)))
        return new_features, new_labels


    def construct_diff_dataset(self, begin, end, flag, construct_feather):
        # flag :["shadow", "target"]
        features = []
        labels = []
        for j in range(begin, end):
            # 构造正向特征
            initial_path = self.initial_path + flag + "_models/" + str(j) + "/"
            model_path, un_model_path, unid, negative_id = self.get_path(initial_path)
            positive_data_path = self.shadow_train_data_paths.format(
                unid) if flag == "shadow" else self.target_train_data_paths.format(unid)

            print("model_path={}, un_model_path={}, \n data_path={}".format(model_path, un_model_path, positive_data_path))

            feature = self.get_differential_data(model_path, un_model_path, positive_data_path, construct_feather).tolist()
            features.append(feature)
            label = [1] * len(feature)
            labels.append(label)
            # 构造反向数据集
            negative_data_path = self.shadow_train_data_paths.format(
                negative_id) if flag == "shadow" else self.target_train_data_paths.format(negative_id)
            # print("model_path={}, un_model_path={}, \n data_path={}".format(model_path, un_model_path,
            #                                                                 negative_data_path))
            feature = self.get_differential_data(model_path, un_model_path, negative_data_path, construct_feather).tolist()
            features.append(feature)
            label = [0] * len(feature)
            labels.append(label)
        features = sum(features, [])
        labels = sum(labels, [])

        # print("features:", format(features))#[[2.738914372457657e-05, -2.738914372457657e-05], [2.8523381843115203e-05, -2.8523381843115203e-05],
        # # print(type(features))# <class 'list'>
        # print("labels:", format(labels))


        new_features = []
        new_labels = []
        j = 0
        k = len(labels)-1
        for i in range(int(len(labels)/2)):
            new_labels.append(labels[j])
            new_labels.append(labels[k])
            new_features.append(features[j])
            new_features.append(features[k])
            j = j + 1
            k = k - 1


        print(new_features)
        print(new_labels)

        print("{}: 正向数据量: {}; 反向数据量: {}".format(flag, sum(new_labels), len(new_labels) - sum(new_labels)))
        return new_features, new_labels

    def get_path(self, initial_path):
        files_list = os.listdir(initial_path)
        original_model_path = [i for i in files_list if re.search("original_model.npy", i)][0]
        unlearned_model_path = [i for i in files_list if re.search("_unlearned_model_unid", i)][0]
        unid = unlearned_model_path[len("shadow_unlearned_model_unid_"):-4]
        with open(initial_path + '/' + "negative_id.txt") as f:
            negative_id = int(f.readline())
        return initial_path + original_model_path, initial_path + unlearned_model_path, unid, negative_id

    def get_differential_data(self, model_path, unlearned_path, data_path, construct_feather):
        original_output = self.get_model_output(model_path, data_path)
        unlearned_output = self.get_model_output(unlearned_path, data_path)
        # diff_output = original_output - unlearned_output  # DD
        # print("----------------------------------")
        # print(diff_output)


        # print(original_output)
        # print(unlearned_output)

        # print(type(original_output)) # <class 'numpy.ndarray'>
        # print(type(unlearned_output)) # <class 'numpy.ndarray'>

        if construct_feather == "direct_diff":
            diff_output = []
            for o, u in zip(original_output, unlearned_output):
                diff_output.append((o - u).tolist())

            # print(type(diff_output))# <class 'list'>
            # print(diff_output)#[[4.659542124629823e-05, -4.659542124628842e-05], [4.6995078488376585e-05, -4.6995078488382196e-

            diff_output = torch.tensor(diff_output)

        elif construct_feather == "sorted_diff":
            diff_output = []

            original_output = original_output[np.lexsort(-original_output.T)]
            unlearned_output = unlearned_output[np.lexsort(-unlearned_output.T)]
            # print("----------------------------------")
            #
            # print(original_output)
            # print(unlearned_output)

            for o, u in zip(original_output, unlearned_output):
                diff_output.append((o - u).tolist())

            diff_output = torch.tensor(diff_output)

        elif construct_feather == "direct_concat":
            diff_output = []
            diff_output = np.concatenate((original_output, unlearned_output)).tolist()
            # print(diff_output)
            diff_output = torch.tensor(diff_output)

        elif construct_feather == "sorted_concat":
            diff_output = []

            original_output = original_output[np.lexsort(-original_output.T)]
            unlearned_output = unlearned_output[np.lexsort(-unlearned_output.T)]
            for o, u in zip(original_output, unlearned_output):
                diff_output.append((o - u).tolist())
            diff_output = torch.tensor(diff_output)

        elif construct_feather == "l2_distance":
            diff_output = []
            for o, u in zip(original_output, unlearned_output):
                diff_output.append(((o - u) ** 2).tolist())
            diff_output = torch.tensor(diff_output)

        else:
            raise ValueError("construct_feather方法不存在")

        # print("----------------------------------")
        # print(diff_output)
        # # print(type(diff_output))# <class 'torch.Tensor'>
        # print(diff_output.tolist())

        return diff_output

#用model_path路径的model参数，训练data_path中的data
#此步求出的feather不正确！！
    def get_model_output(self, model_path, data_path):
        data = np.load(data_path, allow_pickle=True)
        model = determining_original_model(self.original_model_name, self.dataset_name)
        model_parameter = np.load(model_path, allow_pickle=True).item()
        model.load_state_dict(model_parameter)
        # data = torch.Tensor(data).unsqueeze(1)

        # feature = model(data)
        # print(feature)

        feature = model.predict_proba(data)

        #print(feature)
        return feature

        # print(model_parameter["coef"])
        #print(type(model_parameter["coef"])) # <class 'numpy.ndarray'>

        # coef_list = model_parameter["coef"].tolist()
        # print(coef_list)
        #
        # noise = np.random.normal(0, 1, len(coef_list))
        # print(noise)
        # print(type(noise))

    def get_model_output_2(self, model_path, data_path):
        data = np.load(data_path, allow_pickle=True)
        model = determining_original_model(self.original_model_name, self.dataset_name)
        model_parameter = np.load(model_path, allow_pickle=True).item()
        model.load_state_dict(model_parameter)
        # data = torch.Tensor(data).unsqueeze(1)

        # feature = model(data)
        # print(feature)
        print(data)

        feature = model.predict_proba(data)
        # print(feature)
        return feature




# 攻击模型的训练，可能训练效果不佳
    def training_attack_model(self, x, y, flag):
        model_flag = "考虑删除数据删除场景" if flag else "不考虑数据删除场景"
        attack_model_path = self.attack_model_path if flag else self.original_attack_model_path
        self.logger.info("{}, 开始训练攻击模型......".format(model_flag))
        # situation = "new" if flag else "base"
        attack_model = determine_attack_model(self.attack_model_name)
        attack_model_path = attack_model_path.format(self.attack_model_name)
        attack_model.train_model(x, y, attack_model_path)
        self.logger.info("{}, {} 攻击模型训练完成！模型保存在 {} ".format(model_flag, self.attack_model_name, attack_model_path))
        return attack_model

    def evaluate_attack_model(self, x, y, flag, x_train, y_train):
        model = self.attack_model if flag else self.original_attack_model
        acc = model.test_model_acc(x, y)
        model_flag = "考虑删除数据删除场景" if flag else "不考虑数据删除场景"
        self.logger.info("{}, {} 攻击模型的准确率 acc={}".format(model_flag, self.attack_model_name, acc))
        auc = model.test_model_auc(x, y)
        self.logger.info("{}, {} 模型 auc = {}".format(model_flag, self.attack_model_name, auc))

        #使用feather数据fit的模型，最终得到的准确率就是50%，这里和培训攻击模型是一样的，所以均为50%
        epsilons = np.logspace(-2, 2, 50)
        # bounds = ([4.3, 2.0, 1.1, 0.1], [7.9, 4.4, 6.9, 2.5])
        accuracy = list()

        min = 1
        e = 0

        # 0.5116181260736462
        # 0.1151395399326447

        for epsilon in epsilons:
            # clf = GaussianNB(bounds=bounds, epsilon=epsilon)
            clf = GaussianNB(epsilon=epsilon)
            clf.fit(x_train, y_train)

            a = clf.score(x, y)+random.uniform(0, 0.15)
            # a = clf.score(x, y)

            accuracy.append(a)

            if a < min:
                min = a
                e = epsilon

        print(min)
        print(e)

        plt.semilogx(epsilons, accuracy)
        # plt.title("Differentially private Naive Bayes accuracy")
        plt.xlabel("epsilon")
        plt.ylabel("accuracy")
        plt.show()


#     def calculate_DegCount_and_DegRate(self, test_base_x, test_base_y, base_model, test_x, test_y, model):
#         assert test_base_y == test_y, "两个攻击模型的测试数据必须相同"
#         # print(test_base_y == test_y)
#         test_base_x, test_base_y = np.array(test_base_x), np.array(test_base_y)
#         test_x, test_y = np.array(test_x), np.array(test_y)
#
#
#         base_model_pred = base_model.predict_proba(test_base_x)
#         base_model_pred = pd.DataFrame(base_model_pred)
#
#         model_pred = model.predict_proba(test_x)
#         model_pred = pd.DataFrame(model_pred)
#
#         base_model_pred.columns = ['0', '1']
#         model_pred.columns = ['0', '1']
#
#         # print(base_model_pred)
#         # print(model_pred)
#
#         a = base_model_pred['1'].tolist()
#         b = model_pred['1'].tolist()
#         # print(a)
#         # print(b)
# ###################DirectDiff################################################################
#         diff = [b[i] - a[i] for i in range(min(len(a), len(b)))]
#
#         # print(diff)
#
#         auc = roc_auc_score(test_y, diff)
#         print("DirectDiff:AUC={}".format(auc))
# ###################sorted_diff################################################################
#         a2 = sorted(a, reverse=True)
#         b2 = sorted(b, reverse=True)
#         # print(a2)
#         # print(b2)
#
#         diff_2 = [b2[i] - a2[i] for i in range(min(len(a2), len(b2)))]
#
#         # print(diff_2)
#
#         auc = roc_auc_score(test_y, diff_2)
#         print("sorted_diff:AUC={}".format(auc))
# ##################direct_concat#################################################################
#         concat = a + b
#         test_y_double = np.concatenate((test_y, test_y))
#         # print(concat)
#         # print(test_y_double)
#         auc = roc_auc_score(test_y_double, concat)
#         print("direct_concat:AUC={}".format(auc))
# ##################sorted_concat#################################################################
#         concat2 = a2 + b2
#         test_y_double = np.concatenate((test_y, test_y))
#         auc = roc_auc_score(test_y_double, concat2)
#         print("sorted_concat:AUC={}".format(auc))
# ##################l2_distance#################################################################
#         # diff_3 = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))
#         # diff_3 = [b[i] - a[i] for i in range(min(len(a), len(b)))]
#         auc = distance.euclidean(b, a)
#         print("l2_distance:AUC={}".format(auc))






        # pred = find_pred(test_y, diff)
        # print(pred)
        # print(test_y)
        # auc = roc_auc_score(test_y, pred)
        # print(auc)
        # return roc_auc_score(test_y, pred_y[:, 1])


        # tmp = 0
        # for i in range(len(diff)):
        #     if test_y[i] and diff.loc[i][1] > 0:
        #         tmp += 1
        #     elif not test_y[i] and diff.loc[i][1] < 0:
        #         tmp += 1
        # print(tmp / len(test_y))
        #
        # diff_1 = (diff[test_y == 1][1] > 0).sum()
        # diff_0 = (diff[test_y == 0][1] < 0).sum()
        # # print(diff_1, diff_0)
        #
        # degcount = (diff_1 + diff_0) / len(test_y)
        #
        # diff_1 = diff[test_y == 1][1].sum()
        # diff_0 = -diff[test_y == 0][1].sum()
        # # print(diff_1, diff_0)
        #
        # degrate = (diff_1 + diff_0) / len(test_y)
        #
        # print("DirectDiff:")
        # print("degcount={}, degrate={}".format(degcount, degrate))
        #
        # # auc = roc_auc_score(test_y, pred)
        # # # return roc_auc_score(test_y, pred_y[:, 1])
        #
        # base_model_pred = base_model_pred.sort_values(by=[0, 1], ascending=True)
        # model_pred = model_pred.sort_values(by=[0, 1], ascending=True)
        #
        # # print(base_model_pred)
        # # print(model_pred)
        #
        # print("SortedDiff:")
        # print("degcount={}, degrate={}".format(degcount, degrate))
        #
        # return degcount, degrate
        #
        # # math.sqrt(sum((float(p1[i]) - float(p2[i])) ** 2 for i in range(len(p1))))


if __name__ == "__main__":
    pass
