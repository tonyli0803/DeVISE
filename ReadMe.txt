combination_Copy1.py為本次作業的核心中的核心code

為DeVise的本體
也是完全由我們自己打出來的code
而在跑之前，請依照以下步驟，先建置好環境


1.

請至vgg資料夾，跑train_and_val.py
先把vgg的visual model train 好
詳細步驟請至vgg資料夾找ReadMe建置好

2.

請至gensim資料夾，跑train word2vector的程式
詳細步驟請至gensim資料夾找ReadMe建置好

3.

打開
combination_Copy1.py

找到Data_dir = '/home/tony/Desktop/Datasets/cifar_10_batches_binary_version/'
這行code
並將其後面的string，改成你擺cifar_10 image 的檔案路徑
記住請下載binary version 的dataset
可至https://www.cs.toronto.edu/~kriz/cifar.html


4.
輸入python combination_Copy1.py
即可train model

