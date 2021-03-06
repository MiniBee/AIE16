#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: topic_model.py
# @time: 2019/5/28 上午10:25
# @desc:


import numpy as np
import time
import os
import jieba
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier


def get_file_list(dirs):
    file_names = []
    names = []
    for root, dirs, files in os.walk(dirs):
        for file in files:
            file_names.append(root + '/' + file)
            names.append(file.split('_')[0])
    return file_names, names


def get_words(dirs):
    file_names, file_class = get_file_list(dirs)
    words = []
    print('Seging...')

    for itr in file_names:
        with codecs.open(itr, 'r', encoding='utf-8') as file_handle:
            txt = file_handle.read()
            seg_list = jieba.cut(txt, cut_all=False)
            words.append(' '.join(seg_list))
    return words, file_class


def strip_data(sentence, stopwords_file, cut_all=False):
    seg_list = jieba.cut(sentence, cut_all=cut_all)
    words = strip_word(seg_list, stopwords_file)
    return words


def strip_word(seg_list, stopwords_file):
    stopwords = [line.strip() for line in open(stopwords_file, 'r', encoding='utf-8').readlines()]
    words = []
    for word in seg_list:
        if word not in stopwords:
            if word != '\t':
                words.append(word)
    return words


if __name__ == '__main__':
    # test_text = u'你们的好日子到头了'
    # print(strip_data(test_text, 'data/jieba/stopwords'))
    dirs = './data/news'
    data, label = get_words(dirs)
    print(data[:10], label[:10])
    label2int = {}
    int2label = {}
    for idx, itr in enumerate(set(label)):
        label2int[itr] = idx
        int2label[idx] = itr
    label = [label2int[itr] for itr in label]

    data_seg = []
    st = time.perf_counter()
    for itr in data:
        data_seg.append(' '.join(list(strip_data(itr, 'data/jieba/stopwords'))))
    ed = time.perf_counter()
    print('Seg time: ', ed - st)
    print(data_seg[:10])

    tool_chain = []
    method = CountVectorizer(max_df=0.95, min_df=2, max_features=20000)
    st = time.perf_counter()
    data_vect = method.fit_transform(data_seg)
    print('--------------------------')
    print(data_vect)
    ed = time.perf_counter()
    print('Vector time: ', ed - st)
    tool_chain.append(method)

    method = LatentDirichletAllocation(n_components=30, max_iter=5, learning_method='online', learning_offset=50, random_state=0)
    st = time.perf_counter()
    data_dm = method.fit_transform(data_vect)
    ed = time.perf_counter()
    print('LDA time: ', ed - st)
    print(data_dm)
    tool_chain.append(method)

    method = RandomForestClassifier()
    st = time.perf_counter()
    method.fit(data_dm, label)
    ed = time.perf_counter()
    print('RFC time: ', ed - st)

    y = method.predict(data_dm)
    print('RFC acc: ', np.sum(y==label)/len(label))
    tool_chain.append(method)

    text = [
        """
        中国新能源汽车政策:从技术向市场进化版权声明：本文版权为网易汽车所有，转载请注明出处。网易汽车4月17日报道 不久前，中国汽车工业争论由来已久的新能源车技术路线取向上，又有了新动态：不强调某一个技术领域的重要，包容各种技术，总体推进。应该说，这是一个非常市场化的决定，该决定出自工信部、科技部会同国内6大汽车集团的行业内部会议上，可以说涵盖了政策制定者、决策者和执行者三个主体，是个经过广泛讨论而有长远思考的决定。尤其是包容各种技术这点，对于中国千差万别的企业进行的千差万别的新能源车事业来说，可谓“谁都是娘的孩子”，是个皆大欢喜的决定。包容各种路线，让不同的企业利用自己的资源自由去拼争，去找到适合自己技术取向，政府只需做好准入管理，这个逻辑过去曾一度被披上了重复建设的骂名，在包容各种技术路线出台之前，中国汽车企业的新能源路线处于多头准备的状态，等政策的结果，其实已经白白浪费掉了资金和时间。那么，对于一个各种路线都扶持、全线推进这么笼统的决定，为什么会来得这么晚？其实，从1998年中国正式决定发展清洁汽车（当时还没有“新能源车”的说法）到如今，12年过去了，新能源车政策在路线选择上周折不定，在扶持方式上和力度上，不断在探索，总的来说，是雷声大雨点小。因为在这个中国汽车行业经历散乱、低水平竞争的年代，新能源车离人们总是感觉很遥远，政策没有很好的施加载体，只能如空中楼阁。随着公交、出租和私人新能源车补贴试点城市运营推进，新能源车私人补贴政策7月份将出台，政府在加大政策支持力度。国家包容各种路线，每个企业有不同的禀赋和资源条件，但电动车仍是中国很多企业的选择。摇摆不定中探索从政府发出意向不断吆喝到企业纷起迎合，中国新能源车政策探索经历了很长的时间。1998国务院就发出通知，启动了旨在“降低汽车排放污染、促进汽车能源多样化”的“中国清洁汽车行动”，助推燃气汽车顺利实现商业化。此后的2001年，国家在“十五”电动汽车重大专项规划中，确立了“三纵三横”的研发格局。“三纵三横”成为指导当时新能源车发展的红头政策文件，此后的几年里，新能源汽车鲜有宏观的政策，而是一些细小的探索，没有相关标准，汽车企业也在混沌中自己摸索国家会出台什么样的标准。直到2005年，国家质检总局、国家标准委发布了第一批混合动力电动汽车国家标准，在很多方面诸如试验规程、安全、排放测量方法等做了明确规定，此时，混合动力成为当时车企的目标。2007年，发展新能源车已经成为企业共识，很多企业都在向地方政府要资源、要政策发展新能源车，于是出现了一哄而上的态势。该年11月，国家出台了《新能源汽车生产准入规则》提高了新能源汽车的准入门槛。这个门槛首次规定了新能源车概念是包括混合动力汽车、纯电动汽车、燃料电池电动汽车、氢发动机汽车、其他新能源（如高效储能器、二甲醚）的汽车等，但企业需要至少掌握新能源汽车车载能源系统、驱动系统及控制系统三者之一的核心技术。如果说准入规则给企业和行业在新能源路线和概念上提供了新思路，而且规定了进入新能源汽车的必备的要素的话，那么2007年稍后，国家列出目录，才真正明确发展多元化新能源车。2007年12月，新能源车才真正进入发改委的鼓励产业目标，《产业结构调整指导目录（2007年本）》明确提出，鼓励发展节能环保型小排量汽车和使用醇醚燃料、天然气、混合燃料、氢燃料等新型燃料的汽车。可以说，这是“三纵三横”后，新能源路线的又一次扩充。2005年小排量车解禁鼓励发展环保小型车，此前的燃料电池、混合动力和纯电动车的三大取向没有限制政策制定者的思维，车企也没有将自己固定死，加之汽车界内外对新能源政策争论很激烈，尤其争论国家政策应该更加多元，因此，政策的取向明显开始支持多头路线。但是电动车和混合动力仍然是企业们涉足的两大重点。本土企业更倾向于电动车。从笼统扶持到试点补贴新能源车政策不断探索，企业在迎合，巨大的政策和资金扶持效应，对于地方企业来说，是非常惹眼的蛋糕。于是家家点火上新能源项目，被业内诟病为向政府要政策、要资金的形象工程。在长达12年的新能源汽车探索中，车业和行业把技术发明、工程应用当成科学研究和知识创新，“创新”成为企业的主题，很多企业的项目和国家标准上都被外界宣称取得了巨大成果，但是除去私人购买不说，即使公交、出租等行业的新能源车，绝大多数地方还在示范运行，没有很大进展。2008、2009年是新能源政策频出的两年，一个很明显特点是，国家开始大力扶持示范运营，不再在具体的新能源路线上纠缠不休，而是将新能源车市场化作为最为紧迫的任务。2008年5月30日，国家发改委召开了关于汽车行业发展的相关会议，其中将关于汽车新技术以及新能源汽车发展的八类课题交给中国汽车研究中心进行专项研究。此举标志着开始对部分新能源汽车和新技术汽车进行市场化。但是新能源车市场化最重要的，还在于企业自身。尽管政府可以督促地方采购新能源车，但是由于价格很高，大型客车、出租等商用车和私人购买还是无法大力推进，地方采购多数是硬性规定，无法拉动新能源车市场化。于是补贴政策成为推广的一个手段。2009年规定13个城市，今年则增加到20个城市开展能源汽车示范推广试点。鼓励试点城市率先在公交、出租、公务、环卫和邮政等公共服务领域推广使用节能与新能源汽车。私人试点城市也有5个，这些试点城市的最大好处就是得到政府补贴，据悉，私人购买新能源车可能得到最高6万元补助，有望在7月补贴政策出台。目前，新能源车价太高、补贴还无法落实，仍是推广的最大障碍，但这些问题还没法很好解决。随着市场化推广力度加强，企业也意识到新能源需要抱团合作，于是联盟、中外合作等新能源车新的研发推广方式纷纷出现，国家也在不断探索着新能源车标准、鼓励建设充电站等等。去年国家提出，2009～2011年期间，电动汽车要形成50万辆纯电动、充电式混合动力和普通型混合动力等新能源汽车产能，新能源汽车销量占乘用车销售总量的5%左右。尽管多头路线明确，电动车和混合动力还是主导，但按照上述的推进速度，这个目标恐怕还是有点大。
        """,
        """
        人民网记者日本东京11日电（记者陈原）11月11日下午6时，日本东京学习院创立百周年纪念会馆嘉宾云集，备受中日瞩目的中国歌剧《木兰诗篇》在这里成功首演，《木兰诗篇》艺术总监彭丽媛与日本歌唱家芹洋子再次携手同唱日本歌曲《四季歌》。日本皇太子德仁观看了首演。在优美的中国民族音乐旋律中，中国歌剧特有的韵味令人陶醉，木兰故事的深远意境深深打动了观众，日本指挥家堤俊作和中国指挥家李玉宁联合指挥日本皇家管弦乐团演奏，更让演出意味深长。《木兰诗篇》演出结束后，观众激情不减，在持续的掌声中，中国总政歌舞团团长、著名歌唱家、《木兰诗篇》艺术总监彭丽媛走上舞台，与日本歌唱家芹洋子携手同唱日本歌曲《四季歌》。20多年前，彭丽媛作为中国青年代表团成员访日时，曾与芹洋子一起演唱，今日舞台重逢，情深意长，使台上台下的情绪达到最高潮。《木兰诗篇》是中国原创歌剧，取材于木兰替父从军的故事。歌剧《木兰诗篇》分四幕，从“替父从军”、“塞上风云”、|“巾帼情怀”到“和平礼赞”，表现了木兰这个奇女子的宽广心胸，她追求幸福，向往田园，热爱和平。全剧主题歌《木兰花》的最后一句“为教芳香满人间，随风送春向天涯”，更显示出中国人民今日的情怀。歌剧《木兰诗篇》由实力雄厚的总政歌舞团演出，关峡作曲，刘麟编剧，李福祥任总导演。在今天的首场表演中，木兰由雷佳饰演，于爽、姜丽娜、刘旋等主演，其中年纪最小的豆豆来自中国深圳，她和日本小朋友一起主持了演出开场。据介绍，《木兰诗篇》在日本演出时分音乐会版和歌剧版，首场为音乐会版，除东京外，还将在札幌登台。
        """
    ]

    # 分词
    temp = [' '.join(list(jieba.cut(itr))) for itr in text]
    # 使用相同参数进行向量化
    temp = tool_chain[0].transform(temp)
    print(temp)
    # 使用相同参数进行降维
    temp = tool_chain[1].transform(temp)
    print(temp)
    # 使用训练好模型进行预测
    temp = tool_chain[2].predict(temp)
    # 将预测类别转化为标签
    temp = [int2label[itr] for itr in temp]
    print(temp)






