# IAMonSense: Multi-level Handwriting Classification using Spatio-temporal Information

# Updates
> 01 2023: Initial commit

# About
This repository is the implementation of Multi-level Handwriting Classification using Spatio-temporal Information, as described in the following paper:

IAMonSense: Multi-level Handwriting Classification using Spatio-temporal Information, by A. Mustafid, J. Younas, P. Lukowicz, S. Ahmed.

Our paper investigates online handwriting classification which preliminary step for recognition systems and others applications. The problem in this research is, in which level is good for the classification. Each level has benefits and drawbacks. However, there has been little discussion on it and few researchers have addressed the problem because the datasets are also not comparable at all levels. Thus, our research aimed to enhance the datasets and also broaden current knowledge of classification online-handwriting in multi-classes and in multi-levels.

<img src="https://github.com/t4f1d/iamonsense/blob/main/plots/multilevel.png" alt="Data Multilevel" width="600"/>

Without datasets that contain proper data and information at all levels, it is insufficient to do a multi-level online handwriting classification. Consequently, we need to prepare, enhance, and enrich the datasets to be able to perform multi-level classification.

<img src="https://github.com/t4f1d/iamonsense/blob/main/plots/datasets.png" alt="Datasets Comparison" width="600"/>

The contributions of this research establish a foundation and serve as a baseline for systematic and empirical evaluation of online handwriting classification. We provide a new enhancement dataset for the research community, the dataset called `IAMonSense`. It can be used for graph models or deep learning models with different data structure. It contains multi-level information from stroke, word, and line levels. This research has highlighted the importance of line level in the classification problem.


<sub><strong>References:</strong></sub>

<sup>
[14] J. Younas, M. I. Malik, S. Ahmed, F. Shafait, and P. Lukowicz, "Sense the pen: Classification of online handwritten sequences (text, mathematical expression, plot/graph)", Expert Systems with Applications, vol. 172, p. 114 588, 2021. doi: https://doi.org/10.1016/j.eswa.2021.114588.
</sup><br>

<sup>
[27] E. Indermühle, M. Liwicki, and H. Bunke, "IAMonDo-Database: An Online Handwritten Document Database with Non-Uniform Contents", in Proceedings of the 9th IAPR International Workshop on Document Analysis Systems, ser. DAS ’10, Boston, Massachusetts, USA: Association for Computing Machinery, 2010, pp. 97–104. doi: https://doi.org/10.1145/1815330.1815343.
</sup><br>

<sup>
[28] M. Liwicki and H. Bunke, "IAM-OnDB - an On-Line English Sentence Database Acquired from Handwritten Text on a Whiteboard", in Proceedings of the Eighth International Conference on Document Analysis and Recognition, ser. ICDAR ’05, USA: IEEE Computer Society, 2005, pp. 956–961. doi: https://doi.org/10.1109/ICDAR.2005.132.
</sup>


# Dataset
## Download

You can download the dataset from [here (SeaFile)](https://seafile.rlp.net/d/2be24d377f3342ef82ad/) or [here (GDrive)](https://drive.google.com/drive/folders/1RxMVkQiNu5fh-R9TZeI_ez_jQv27Adxu?usp=share_link).

## Structure
The structure of the `IAMonSense` dataset,

```shell
IAMonSense/
├─ SenseThePen+/
│  └─ line_data/
│     ├─ p1/
│     │  ├─ l_1.csv
│     │  ├─ l_2.csv
│     │  ├─ l_3.csv
│     │  ├─ ...
│     │  └─ l_21.csv
│     ├─ p2/
│     │  ├─ l_1.csv
│     │  ├─ l_2.csv
│     │  ├─ l_3.csv
│     │  ├─ ...
│     │  └─ l_72.csv
│     ├─ .../
│     └─ p20/
│        ├─ l_1.csv
│        ├─ l_2.csv
│        ├─ l_3.csv
│        ├─ ...
│        └─ l_83.csv
│
├─ IAMonDo+/
│  ├─ 001.csv
│  ├─ 001a.csv
│  ├─ 001b.csv
│  ├─ 001c.csv
│  ├─ 001d.csv
│  ├─ 001e.csv
│  ├─ ...
│  └─ 982.csv
│
├─ IAM-OnDB+/
│  └─ line_data/
│     ├─ a01-000/
│     │  ├─ a01-000u-01.csv
│     │  ├─ a01-000u-02.csv
│     │  ├─ a01-000u-03.csv
│     │  ├─ ...
│     │  └─ a01-000u-06.csv
│     ├─ a01-001/
│     │  ├─ a01-001w-01.csv
│     │  ├─ a01-001w-02.csv
│     │  ├─ a01-001w-03.csv
│     │  ├─ ...
│     │  └─ a01-001z-09.csv
│     ├─ .../
│     └─ z01-000/
│        ├─ z01-000-01.csv
│        ├─ z01-000-02.csv
│        ├─ z01-000-03.csv
│        ├─ ...
│        └─ z01-000z-08.csv
│
├─ SenseThePen_train.csv
├─ SenseThePen_val.csv
├─ SenseThePen_test.csv
├─ IAMonDo_train.csv
├─ IAMonDo_val.csv
├─ IAMonDo_test.csv
├─ IAM-OnDB_train.csv
├─ IAM-OnDB_val.csv
└─ IAM-OnDB_test.csv
```


## Statistics

| Datasets    | # of files  | # of strokes| # of words  | # of lines  | # of classes|
| ----------- | ----------: | ----------: | ----------: | ----------: | ----------: |
| SenseThePen+| 1,595       | 36,329      | 12,947      | 1,595       | 3           |
| IAMonDo+    | 941         | 356,189     | 87,924      | 18,658      | 3           |
| IAM-OnDB+   | 12,190      | 304,696     | 64,084      | 12,190      | 1           |


# Results

## Deep Learning Model
![Performance Analysis Deep Learning Model](https://github.com/t4f1d/iamonsense/blob/main/plots/result1.png)

## Graph Model
![Performance Analysis Graph Model](https://github.com/t4f1d/iamonsense/blob/main/plots/result2.png)

## Transformer Model
![Performance Analysis Transformer Model](https://github.com/t4f1d/iamonsense/blob/main/plots/result3.png)

## State-of-the-art Comparison
![State-of-the-art Comparison](https://github.com/t4f1d/iamonsense/blob/main/plots/result4.png)


<sub><strong>References:</strong></sub>

<sup>
[10]  J.-Y. Ye, Y.-M. Zhang, Q. Yang, and C.-L. Liu, “Contextual stroke classification in online handwritten documents with edge graph attention networks”, SN Computer Science, vol. 1, no. 3, pp. 1–13, 2020. doi: https://doi.org/10.1007/s42979-020-00177-0.
</sup><br>

<sup>
[14]  J. Younas, M. I. Malik, S. Ahmed, F. Shafait, and P. Lukowicz, “Sense the pen: Classification of online handwritten sequences (text, mathematical expression, plot/graph)”, Expert Systems with Applications, vol. 172, p. 114 588, 2021. doi: https://doi.org/10.1016/j.eswa.2021.114588.
</sup><br>

<sup>
[15]  J. Younas, S. Fritsch, G. Pirkl, S. Ahmed, M. I. Malik, F. Shafait, and P. Lukowicz, “What Am I Writing: Classification of On-Line Handwritten Sequences.”, in Intelligent Environments (Workshops), ser. Ambient Intelligence and Smart Environments, vol. 23, IOS Press, 2018, pp. 417–426.
</sup><br>

<sup>
[34]  I. Degtyarenko, I. Deriuga, A. Grygoriev, S. Polotskyi, V. Melnyk, D. Zakharchuk, and O. Radyvonenko, “Hierarchical Recurrent Neural Network for Handwritten Strokes Classification”, in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 2865–2869. doi: https://doi.org/10.1109/ICASSP39728.2021.9413412.
</sup>



## License
This project is licensed under the MIT License. See LICENSE for more details.


## Citations
```shell
@article{iamonsense,
    Title={IAMonSense: Multi-level Handwriting Classification using Spatio-temporal Information},
    Author={Mustafid, Ahmad and Younas, Junaid and Lukowicz, Paul and Ahmed, Sheraz},
    DOI={10.21203/rs.3.rs-2275927/v1},
    Publisher={Research Square},
    Year={2022},
    URL={https://doi.org/10.21203/rs.3.rs-2275927/v1},
}
```

## Acknowledgements
We would like to thank the following people for their support: Siti Helmiyah, M. Murah Pamuji, Boby Gunarso, Noor Titan Putri Hartono, Sukma Dyah Aini. Special thanks to RPTU and DFKI.
