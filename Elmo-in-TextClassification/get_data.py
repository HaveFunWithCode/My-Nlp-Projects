import math
import os
import time
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np
import h5py

stops='''
ای
ها
های
روز
کن
باش
نکن
داری
دارد
باشه
و
در
به
از
که
این
را
با
است
برای
آن
یک
خود
تا
کرد
بر
هم
نیز
گفت
می‌شود
وی
شد
دارد
ما
اما
یا
شده
باید
هر
آنها
بود
او
دیگر
دو
مورد
می‌کند
شود
کند
وجود
بین
پیش
شده_است
پس
نظر
اگر
همه
یکی
حال
هستند
من
کنند
نیست
باشد
چه
بی
می
بخش
می‌کنند
همین
افزود
هایی
دارند
راه
همچنین
روی
داد
بیشتر
بسیار
سه
داشت
چند
سوی
تنها
هیچ
میان
اینکه
شدن
بعد
جدید
ولی
حتی
کردن
برخی
کردند
می‌دهد
اول
نه
کرده_است
نسبت
بیش
شما
چنین
طور
افراد
تمام
درباره
بار
بسیاری
می‌تواند
کرده
چون
ندارد
دوم
بزرگ
طی
حدود
همان
بدون
البته
آنان
می‌گوید
دیگری
خواهد_شد
کنیم
قابل
یعنی
رشد
می‌توان
وارد
کل
ویژه
قبل
براساس
نیاز
گذاری
هنوز
لازم
سازی
بوده_است
چرا
می‌شوند
وقتی
گرفت
کم
جای
حالی
تغییر
پیدا
اکنون
تحت
باعث
مدت
فقط
زیادی
تعداد
آیا
بیان
رو
شدند
عدم
کرده_اند
بودن
نوع
بلکه
جاری
دهد
برابر
مهم
بوده
اخیر
مربوط
امر
زیر
گیری
شاید
خصوص
آقای
اثر
کننده
بودند
فکر
کنار
اولین
سوم
سایر
کنید
ضمن
مانند
باز
می‌گیرد
ممکن
حل
دارای
پی
مثل
می‌رسد
اجرا
دور
منظور
کسی
موجب
طول
امکان
آنچه
تعیین
گفته
شوند
جمع
خیلی
علاوه
گونه
تاکنون
رسید
ساله
گرفته
شده_اند
علت
چهار
داشته_باشد
خواهد_بود
طرف
تهیه
تبدیل
مناسب
زیرا
مشخص
می‌توانند
نزدیک
جریان
روند
بنابراین
می‌دهند
یافت
نخستین
بالا
پنج
ریزی
عالی
چیزی
نخست
بیشتری
ترتیب
شده_بود
خاص
خوبی
خوب
شروع
فرد
کامل
غیر
می‌رود
دهند
آخرین
دادن
جدی
بهترین
شامل
گیرد
بخشی
باشند
تمامی
بهتر
داده_است
حد
نبود
کسانی
می‌کرد
داریم
علیه
می‌باشد
دانست
ناشی
داشتند
دهه
می‌شد
ایشان
آنجا
گرفته_است
دچار
می‌آید
لحاظ
آنکه
داده
بعضی
هستیم
اند
برداری
نباید
می‌کنیم
نشست
سهم
همیشه
آمد
اش
وگو
می‌کنم
حداقل
طبق
جا
خواهد_کرد
نوعی
چگونه
رفت
هنگام
فوق
روش
ندارند
سعی
بندی
شمار
کلی
کافی
مواجه
همچنان
زیاد
سمت
کوچک
داشته_است
چیز
پشت
آورد
حالا
روبه
سال‌های
دادند
می‌کردند
عهده
نیمه
جایی
دیگران
سی
بروز
یکدیگر
آمده_است
جز
کنم
سپس
کنندگان
خودش
همواره
یافته
شان
صرف
نمی‌شود
رسیدن
چهارم
یابد
متر
ساز
داشته
کرده_بود
باره
نحوه
کردم
تو
شخصی
داشته_باشند
محسوب
پخش
کمی
متفاوت
سراسر
کاملا
داشتن
نظیر
آمده
گروهی
فردی
ع
همچون
خطر
خویش
کدام
دسته
سبب
عین
آوری
متاسفانه
بیرون
دار
ابتدا
شش
افرادی
می‌گویند
سالهای
درون
نیستند
یافته_است
پر
خاطرنشان
گاه
جمعی
اغلب
دوباره
می‌یابد
لذا
زاده
گردد
اینجا

'''

import preprocessor as p

def get_sentifasttextdatset():


    fasttext_vectors=pd.HDFStore('vecs/senti-fasttext/senti_fasttext_vecs.h5')
    cbow_vectores=fasttext_vectors['cbow']
    skip_vectores=fasttext_vectors['skip']


    # --------------------load y and main texts------------------------
    data = pd.read_csv('data/sentiment.csv')

    x = list(data['t'])
    y = list(data[' c'])
    y = [int(yi) if not math.isnan(yi) else 4 for yi in y]

    le = preprocessing.LabelEncoder()
    le.fit(y)

    def encode(le, labels):
        enc = le.transform(labels)
        return tf.keras.utils.to_categorical(enc)

    x_enc = x
    y_enc = encode(le, y)

    return cbow_vectores,skip_vectores,y_enc,300,x_enc

def getfasttextdatset():


    fasttext_vectors=pd.HDFStore('vecs/tweet-fasttext/insta_fasttext_vecs.h5')
    cbow_vectores=fasttext_vectors['cbow']
    skip_vectores=fasttext_vectors['skip']


    # --------------------load y and main texts------------------------
    data = pd.read_csv('data/tweet_labeled.csv')

    x = list(data['text'])
    y = list(data['cat'])

    le = preprocessing.LabelEncoder()
    le.fit(y)

    def encode(le, labels):
        enc = le.transform(labels)
        return tf.keras.utils.to_categorical(enc)

    x_enc = x
    y_enc = encode(le, y)

    return cbow_vectores,skip_vectores,y_enc,300,x_enc

def getdataset(vector=0,sequence_embeding_method='avg',_max_sent_length=100):
    '''

    :param vector: 0:word embedding|1:lstm_1|2:lstm_2|3 avarage |4 avg lstm1 , lstm2
    :return:
    '''
    # -------------------------------------------------------
    # load dataset


    data = pd.read_csv('data/tweet_labeled.csv')

    x = list(data['text'])
    # preprocees
    p.set_options(p.OPT.URL, p.OPT.MENTION)

    y= list(data['cat'])

    cat_number=len(list(set(y)))
    ss=stops.split('\n')
    sample_num=len(x)
    for i,text in enumerate(x):
        x[i]=p.clean(text).replace('RT :', '').replace('\n',' ')

    le = preprocessing.LabelEncoder()
    le.fit(y)

    def encode(le, labels):
        enc = le.transform(labels)
        return tf.keras.utils.to_categorical(enc)

    x_enc = x
    y_enc = encode(le, y)

    # -------------------------------------------------------
    embedding_file = 'vecs/elmo/elmo_embeddings.hdf5'
    # embedding_matrix=[]

    # load  embedding weights
    elmo_embeddings = []

    with h5py.File(embedding_file, 'r') as elmo_weights:
        #     based on evaluation paper
        if sequence_embeding_method=='avg':
            # avarage method
            for i in range(0, sample_num):
                print(str(i))
                if vector in [0, 1, 2]:
                    elmo_embeddings.append(np.mean(elmo_weights[str(i)].value[vector][:], axis=0))
                elif vector in [3]:
                    sent_embedding = elmo_weights[str(i)]

                    num_words = sent_embedding.value.shape[1]
                    words_embd = []
                    print(num_words)
                    for j in range(num_words):
                        words_embd.append(np.mean(
                            [sent_embedding.value[0][j], sent_embedding.value[1][j], sent_embedding.value[2][j]],
                            axis=0))
                    elmo_embeddings.append(np.mean(words_embd, axis=0))
                elif vector in [4]:
                    avarage_embedding = []
                    sent_embedding = elmo_weights[str(i)]

                    num_words = sent_embedding.value.shape[1]
                    words_embd = []
                    print(num_words)
                    for j in range(num_words):
                        words_embd.append(np.mean(
                            [sent_embedding.value[1][j], sent_embedding.value[2][j]],
                                    axis=0))
                    elmo_embeddings.append(np.mean(words_embd, axis=0))
            embedding_length=1024
        #     based on ulmif (Concat pooling)
        elif sequence_embeding_method=='concat':
            # my method
            for i in range(0, sample_num):
                print(str(i))
                doc_embedding =np.concatenate((np.mean(elmo_weights[str(i)].value[0][:], axis=0),
                                 np.mean(elmo_weights[str(i)].value[1][:], axis=0),
                                 np.mean(elmo_weights[str(i)].value[2][:], axis=0)))

                elmo_embeddings.append(doc_embedding)


            embedding_length=3072
        elif sequence_embeding_method=='mymethod2':
            max_sent_length=_max_sent_length
            # my method
            for i in range(0, sample_num):
                print(str(i))
                sent_embedding = elmo_weights[str(i)]

                num_words = sent_embedding.value.shape[1]
                words_embd = []
                print(num_words)
                for j in range(num_words):
                    words_embd.append(np.mean(
                        [sent_embedding.value[0][j], sent_embedding.value[1][j], sent_embedding.value[2][j]],
                        axis=0))
                doc_embedding=np.concatenate((words_embd))
                if num_words>max_sent_length:
                    print('error')
                elif num_words<max_sent_length:
                    doc_embedding=np.concatenate((doc_embedding,np.zeros((max_sent_length-num_words)*1024)))
                elmo_embeddings.append(doc_embedding)
            embedding_length = 1024 * max_sent_length
        elif sequence_embeding_method=='mymethod2pca':
            max_sent_length = 100
            elmo_embeddings=np.zeros(shape=(sample_num,max_sent_length*1024),dtype=float)

            # my method
            for i in range(0, sample_num):
                print(str(i))
                sent_embedding = elmo_weights[str(i)]

                num_words = sent_embedding.value.shape[1]
                words_embd = []
                print(num_words)
                for j in range(num_words):
                    words_embd.append(np.mean(
                        [sent_embedding.value[0][j], sent_embedding.value[1][j], sent_embedding.value[2][j]],
                        axis=0))
                doc_embedding=np.concatenate((words_embd))
                if num_words>max_sent_length:
                    print('error')
                elif num_words<max_sent_length:
                    doc_embedding=np.concatenate((doc_embedding,np.zeros((max_sent_length-num_words)*1024)))
                elmo_embeddings[i]=doc_embedding
            embedding_length = 1024 * max_sent_length
        elif sequence_embeding_method == 'concatpooling':


            for i in range(0, sample_num):
                print(str(i))
                sent_embedding = elmo_weights[str(i)]

                num_words = sent_embedding.value.shape[1]
                words_embd = []
                print(num_words)
                for j in range(num_words):
                    words_embd.append(np.concatenate((
                        np.mean([sent_embedding.value[1][j], sent_embedding.value[2][j]], axis=0),
                        np.max([sent_embedding.value[1][j], sent_embedding.value[2][j]], axis=0),
                        sent_embedding.value[2][j]

                    )))
                doc_embedding = np.concatenate((words_embd))
                if num_words > 100:
                    print('error')
                elif num_words < 100:
                    doc_embedding = np.concatenate((doc_embedding, np.zeros((100 - num_words) * 1024)))
                elmo_embeddings.append(doc_embedding)

            embedding_length=3072*100
        elif sequence_embeding_method == 'mymethod4':


            for i in range(0, sample_num):
                print(str(i))
                sent_embedding = elmo_weights[str(i)]

                num_words = sent_embedding.value.shape[1]
                words_embd = []
                print(num_words)
                for j in range(num_words):
                    words_embd.append(np.concatenate((
                        np.mean([sent_embedding.value[1][j], sent_embedding.value[2][j]], axis=0),
                        np.max([sent_embedding.value[1][j], sent_embedding.value[2][j]], axis=0),
                        sent_embedding.value[2][j]

                    )))
                doc_embedding = np.mean(words_embd,axis=0)

                elmo_embeddings.append(doc_embedding)

            embedding_length=3072



    return elmo_embeddings,y_enc,embedding_length,cat_number,x_enc

def getsentdataset(vector=0,sequence_embeding_method='avg',_max_sent_length=100):
    '''


    '''
    # -------------------------------------------------------
    # load dataset


    data = pd.read_csv('data/sentiment.csv')

    x = list(data['t'])
    # preprocees
    p.set_options(p.OPT.URL, p.OPT.MENTION)

    y= list(data[' c'])
    y=[int(yi) if not math.isnan(yi) else 4 for yi in y]

    cat_number=len(list(set(y)))
    ss=stops.split('\n')
    sample_num=len(x)
    for i,text in enumerate(x):
        x[i]=p.clean(text).replace('RT :', '').replace('\n',' ')

    le = preprocessing.LabelEncoder()
    le.fit(y)

    def encode(le, labels):
        enc = le.transform(labels)
        return tf.keras.utils.to_categorical(enc)

    x_enc = x
    y_enc = encode(le, y)

    # -------------------------------------------------------
    embedding_file = 'vecs/elmosent/elmo_embeddings_sentiment.hdf5'
    # embedding_matrix=[]

    # load  embedding weights
    elmo_embeddings = []

    with h5py.File(embedding_file, 'r') as elmo_weights:
        #     based on evaluation paper
        if sequence_embeding_method=='avg':
            # avarage method
            for i in range(0, sample_num):
                print(str(i))
                if vector in [0, 1, 2]:
                    elmo_embeddings.append(np.mean(elmo_weights[str(i)].value[vector][:], axis=0))
                elif vector in [3]:
                    sent_embedding = elmo_weights[str(i)]

                    num_words = sent_embedding.value.shape[1]
                    words_embd = []
                    print(num_words)
                    for j in range(num_words):
                        words_embd.append(np.mean(
                            [sent_embedding.value[0][j], sent_embedding.value[1][j], sent_embedding.value[2][j]],
                            axis=0))
                    elmo_embeddings.append(np.mean(words_embd, axis=0))
                elif vector in [4]:
                    avarage_embedding = []
                    sent_embedding = elmo_weights[str(i)]

                    num_words = sent_embedding.value.shape[1]
                    words_embd = []
                    print(num_words)
                    for j in range(num_words):
                        words_embd.append(np.mean(
                            [sent_embedding.value[1][j], sent_embedding.value[2][j]],
                                    axis=0))
                    elmo_embeddings.append(np.mean(words_embd, axis=0))
            embedding_length=1024
        #     based on ulmif (Concat pooling)
        elif sequence_embeding_method=='concat':
            # my method
            for i in range(0, sample_num):
                print(str(i))
                doc_embedding =np.concatenate((np.mean(elmo_weights[str(i)].value[0][:], axis=0),
                                 np.mean(elmo_weights[str(i)].value[1][:], axis=0),
                                 np.mean(elmo_weights[str(i)].value[2][:], axis=0)))

                elmo_embeddings.append(doc_embedding)


            embedding_length=3072
        elif sequence_embeding_method=='mymethod2':
            max_sent_length=_max_sent_length
            # my method
            for i in range(0, sample_num):
                print(str(i))
                sent_embedding = elmo_weights[str(i)]

                num_words = sent_embedding.value.shape[1]
                words_embd = []
                print(num_words)
                for j in range(num_words):
                    words_embd.append(np.mean(
                        [sent_embedding.value[0][j], sent_embedding.value[1][j], sent_embedding.value[2][j]],
                        axis=0))
                doc_embedding=np.concatenate((words_embd))
                if num_words>max_sent_length:
                    print('error')
                elif num_words<max_sent_length:
                    doc_embedding=np.concatenate((doc_embedding,np.zeros((max_sent_length-num_words)*1024)))
                elmo_embeddings.append(doc_embedding)
            embedding_length = 1024 * max_sent_length
        elif sequence_embeding_method=='mymethod2pca':
            max_sent_length = 100
            elmo_embeddings=np.zeros(shape=(sample_num,max_sent_length*1024),dtype=float)

            # my method
            for i in range(0, sample_num):
                print(str(i))
                sent_embedding = elmo_weights[str(i)]

                num_words = sent_embedding.value.shape[1]
                words_embd = []
                print(num_words)
                for j in range(num_words):
                    words_embd.append(np.mean(
                        [sent_embedding.value[0][j], sent_embedding.value[1][j], sent_embedding.value[2][j]],
                        axis=0))
                doc_embedding=np.concatenate((words_embd))
                if num_words>max_sent_length:
                    print('error')
                elif num_words<max_sent_length:
                    doc_embedding=np.concatenate((doc_embedding,np.zeros((max_sent_length-num_words)*1024)))
                elmo_embeddings[i]=doc_embedding
            embedding_length = 1024 * max_sent_length
        elif sequence_embeding_method == 'concatpooling':


            for i in range(0, sample_num):
                print(str(i))
                sent_embedding = elmo_weights[str(i)]

                num_words = sent_embedding.value.shape[1]
                words_embd = []
                print(num_words)
                for j in range(num_words):
                    words_embd.append(np.concatenate((
                        np.mean([sent_embedding.value[1][j], sent_embedding.value[2][j]], axis=0),
                        np.max([sent_embedding.value[1][j], sent_embedding.value[2][j]], axis=0),
                        sent_embedding.value[2][j]

                    )))
                doc_embedding = np.concatenate((words_embd))
                if num_words > 100:
                    print('error')
                elif num_words < 100:
                    doc_embedding = np.concatenate((doc_embedding, np.zeros((100 - num_words) * 1024)))
                elmo_embeddings.append(doc_embedding)

            embedding_length=3072*100
        elif sequence_embeding_method == 'mymethod4':


            for i in range(0, sample_num):
                print(str(i))
                sent_embedding = elmo_weights[str(i)]

                num_words = sent_embedding.value.shape[1]
                words_embd = []
                print(num_words)
                for j in range(num_words):
                    words_embd.append(np.concatenate((
                        np.mean([sent_embedding.value[1][j], sent_embedding.value[2][j]], axis=0),
                        np.max([sent_embedding.value[1][j], sent_embedding.value[2][j]], axis=0),
                        sent_embedding.value[2][j]

                    )))
                doc_embedding = np.mean(words_embd,axis=0)

                elmo_embeddings.append(doc_embedding)

            embedding_length=3072



    return elmo_embeddings,y_enc,embedding_length,cat_number,x_enc
