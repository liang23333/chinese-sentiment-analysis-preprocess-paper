# chinese-sentiment--analysis-preprocess
2017-4-23:
====  
   今天主要是真正的开始，我爬取了网络热词，爬取了符号表情，然后做成一个dict，方便以后处理时使用。


2017-4-24：
====  

  18：37
  -------  

    数据集真正的找到完全符合我们的要求的确实是不现实的，因此只能手动添加噪声。我看了一下计划书，要添加的包括以下几部分：
符号表情，网络热词，还有将一些词汇改成英语。今天早上上课的时候，我改了一会，不过确实太多了，三种噪声都需要修改，所以我将符号表情分成了pos和neg两部分，
然后分别往pos的数据集里加入pos的符号表情，neg的数据集里加入neg的符号表情，用代码的形式实现，直接在文件末尾随机选择一个符号表情加入，不过因为是噪声，
所以做了一个随机数处理，大约有一半的数据集做了处理。网络热词也是如此。不过英语词汇没别的办法，只能这样，自己慢慢加。


   20：37
   -------  
         将消极数据集里的部分数据手工由中文改成了英文，大约改了200个。
  
  
2017-4-25：
====
   
   18：34
   ------- 
   
         今天将积极数据集里的部分单词改成了英文，大概1000总数据集，164里经过我的改动。
   
   
   20：54
   ------- 
   
         加入了重复字噪声。到这为止，数据集正式准备完毕。
     
   
   
   21：40
   ------- 
   
         什么预处理都不做，直接用tfidf生成向量，用NB生成的结果，交叉验证结果为0.89，在测试集中accurancy为0.76，结果还不错
     
2017-4-26：
====

   14：55
   ------- 
          昨天的是character level,昨天忘记分词改成word level了，今天改成word level,结果精确度到了0.92   有点太高了
      
   21：37
   ------- 
         finish the task of removing links,replacing repeatwords,removing stopwords,removing number,removing punctuation,just the same as english preprocess
      
      
2017-4-27：
====

   14：55
   -------
         add links to dataset,and finish all preprocess operation functions.



2017-5-2：
====

   19：00
   -------
         前几天由于弄CNN和LSTM，需要在服务器上做，就没有更新，但是因为对github不熟悉，导致那个文档坏了，我就新建了一个，目前来说各个部分都已经完成，8个operation  4个分类器  各个步骤全部完成   下一步就开始真正的实验部分
<<<<<<< HEAD


 
2017-5-6：
====
   15：00
   -------
         完成了中文的全部实验部分，为了让结果更显著一些,决定再做英语的预处理部分作比较。


2017-5-10：
====
   22：41
   -------
         完成了英文的全部实验部分，下一步开始写论文。
=======
         
         
