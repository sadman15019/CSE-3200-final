### Blood Pressure Measurement Based on PPG Extracted from Fingertip Video
----

**python implementation**

```python
Version: 0.0.1  
Author : Md. Rezwanul Haque,
         S.M. Taslim Uddin Raju
```
### **Related resources**:


**LOCAL ENVIRONMENT**  
```python
OS          : Ubuntu 22.04 LTS       
Memory      : 8.0 GiB 
Processor   : Intel® Core™ i5-5200U CPU @ 2.20GHz × 4    
Graphics    : AMD® Radeon r5 m255 / Mesa Intel® HD Graphics 5500 (BDW GT2)  
Gnome       : 42.1 
```

**python requirements**
* **pip requirements**: ```pip install -r requirements.txt``` 
> Its better to use a virtual environment 
OR use conda-
* **conda**: use environment.yml: ```conda env create -f environment.yml```


# dataset requirements

create a dataset as follows:
 
```
dataset_folder
    |-raw_videos
        |-sample_1.mp4
        |-2.mp4
        |-sample-3.mp4
        |-...........
        |-...........

    |-data.csv

```

```
* data.csv colums:

    * SL              : Serial No.
    * Patient's ID    : ID of Subjects (like, 0001, 0901, ...)
    * Name            : Name of subjects (xyz, abcd, ...)
    * Age             : Age of subjects (12,69, ...)
    * Sex(M/F)        : Male or Female (M/F)
    * File_ext(*.mp4) : video file extension (.mp4)
    * Hb (g/dL)       : Hemoglobin concentration
    * Gl (mmol/L)     : Glucose concentration
    * Cr (ml/dl)      : Creatinine concentration
```

# Execution
- ```conda activate my_env```
- ```cd scripts```
- run: ```./server.sh```


<!-- **LaTex Utils Install**

```sudo apt install texlive-latex-base```

```sudo apt-get install texlive-latex-extra```

# Execution
- ```conda activate your_env```
- ```cd scripts```
- run: ```./server.sh```


- use **debug.ipynb** for visualization -->

---
## ABOUT
### Refereneces:

**Cite:** Please cite the following papers. If you find this code useful in your research, please consider citing.

```bibtext
@article{haque2021novel,
  title={A novel technique for non-invasive measurement of human blood component levels from fingertip video using DNN based models},
  author={Haque, Md Rezwanul and Raju, SM Taslim Uddin and Golap, Md Asaf-Uddowla and Hashem, MMA},
  journal={IEEE Access},
  volume={9},
  pages={19025--19042},
  year={2021},
  publisher={IEEE}
}
```

```bibtext
@article{haque2021corrections,
  title={Corrections to" A Novel Technique for Non-Invasive Measurement of Human Blood Component Levels From Fingertip Video Using DNN Based Models".},
  author={Haque, MD Rezwanul and Raju, SM Taslim Uddin and Golap, MD Asaf-Uddowla and Hashem, MMA},
  journal={IEEE Access},
  volume={9},
  pages={84178--84179},
  year={2021}
}
```

```bibtext
@article{golap2021hemoglobin,
  title={Hemoglobin and glucose level estimation from PPG characteristics features of fingertip video using MGGP-based model},
  author={Golap, Md Asaf-uddowla and Raju, SM Taslim Uddin and Haque, Md Rezwanul and Hashem, MMA},
  journal={Biomedical Signal Processing and Control},
  volume={67},
  pages={102478},
  year={2021},
  publisher={Elsevier}
}
```

```bibtext
@inproceedings{golap2019non,
  title={Non-invasive hemoglobin concentration measurement using MGGP-based model},
  author={Golap, Md Asaf-uddowla and Hashem, MMA},
  booktitle={2019 5th International Conference on Advances in Electrical Engineering (ICAEE)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```