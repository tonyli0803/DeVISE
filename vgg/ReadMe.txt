本code參照
https://github.com/kevin28520/My-TensorFlow-tutorials
所建成



建置環境步驟如下

1.

請至https://www.cs.toronto.edu/~kriz/cifar.html下載binary version 的cifar10 dataset並解壓縮

2.

打開tran_and_val.ipynb或是training_and_val.py
找到data_dir = '/home/tony/Desktop/Datasets/cifar_10_batches_binary_version/'
這行code
並將其後面的string，改成你擺cifar_10 image 的檔案路徑
注意是下載binary version 的dataset


3.

可以至
https://github.com/kevin28520/My-TensorFlow-tutorials/tree/master/04%20VGG%20Tensorflow
去下載vgg.npy
此為pretrain weight，可以加速訓練過程，較快達到好的效果
並放入vgg16_pretrain資料夾中


4.
輸入python tran_and_val.py
或是使用jupyter notebook跑training_and_val.ipynb
即可把visual model 的部分建置完畢

建置完後請至上一層的gensim資料夾繼續建置W2V MODEL