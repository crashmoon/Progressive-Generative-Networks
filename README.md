# Progressive-Generative-Networks
Haoran Zhang, Zhenzhen Hu, Changzhi Luo, Wangmeng Zuo, Meng Wang:
Semantic Image Inpainting with Progressive Generative Networks. ACM Multimedia 2018: 1939-1947
https://dl.acm.org/citation.cfm?doid=3240508.3240625

@inproceedings{DBLP:conf/mm/ZhangHLZW18,
  author    = {Haoran Zhang and
               Zhenzhen Hu and
               Changzhi Luo and
               Wangmeng Zuo and
               Meng Wang},
  title     = {Semantic Image Inpainting with Progressive Generative Networks},
  booktitle = {2018 {ACM} Multimedia Conference on Multimedia Conference, {MM} 2018,
               Seoul, Republic of Korea, October 22-26, 2018},
  pages     = {1939--1947},
  year      = {2018},
  crossref  = {DBLP:conf/mm/2018},
  url       = {https://doi.org/10.1145/3240508.3240625},
  doi       = {10.1145/3240508.3240625},
  timestamp = {Wed, 21 Nov 2018 12:44:21 +0100},
  biburl    = {https://dblp.org/rec/bib/conf/mm/ZhangHLZW18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

# Prerequisites

Python, NumPy, SciPy, Matplotlib

A recent NVIDIA GPU

**A latest master version of Pytorch**


# Progress
1.Make a CSV file with make_images.py

2.Modify the parameters in gan_lstm.py (Param.out_path)

3.python gan_lstm.py (run the program)

# Results
![p0](imgs/pgn/test_image_0.jpg)
![p1](imgs/pgn/test_image_1.jpg)
![p2](imgs/pgn/test_image_2.jpg)
![p3](imgs/pgn/test_image_3.jpg)
![p4](imgs/pgn/test_image_4.jpg)
![p5](imgs/pgn/test_image_5.jpg)

![x0](imgs/pgn/imagenet_test_image_0.jpg)
![x1](imgs/pgn/imagenet_test_image_1.jpg)
![x2](imgs/pgn/imagenet_test_image_2.jpg)
![x3](imgs/pgn/imagenet_test_image_3.jpg)


