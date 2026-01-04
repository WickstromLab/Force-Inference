# Force-Inference for epithelial cells

`@Author: Ali Hashmi`

see my Wolfram Community post for more information: https://community.wolfram.com/groups/-/m/t/1571507

This repository contains implementations of the `Bayesian Force Inference` technique by Ishihara and Sugimura (Journal of Theoretical Biology, 2012). The scripts infer tension for edges between epithelial cells and the pressure within the cells. The input is a binarized image of an epithelia.

## Implementations

### Mathematica Implementation (Original)
The script files/notebooks located in the `main` folder contain the original Mathematica implementation.

### Python Implementation (New!)
A complete Python implementation is now available:
- **Main script:** `force_inference.py`
- **Documentation:** `PYTHON_README.md`
- **Example usage:** `example_usage.py`
- **Dependencies:** `requirements.txt`

**Quick Start (Python):**
```bash
pip install -r requirements.txt
python force_inference.py "for ReadMe/image.tif" True
```

See `PYTHON_README.md` for detailed Python usage instructions.

##### Please note that the borders of the image used below, `image.tif` (inside 'for ReadMe' folder), have pixels that are filled. 

https://www.biorxiv.org/content/early/2018/11/23/475012 (for more details, check recent work by Lenne Lab)



![alt-text](https://github.com/alihashmiii/Force-Inference/blob/master/for%20ReadMe/im1.png)


![alt-text](https://github.com/alihashmiii/Force-Inference/blob/master/for%20ReadMe/im2.png)


![alt-text](https://github.com/alihashmiii/Force-Inference/blob/master/for%20ReadMe/im3.png)


![alt-text](https://github.com/alihashmiii/Force-Inference/blob/master/for%20ReadMe/im4.png)



#### Robustness Check
<a href="https://github.com/alihashmiii/Force-Inference/blob/master/miscellaneous/robustness%20check.pdf"><img src= "https://github.com/alihashmiii/Force-Inference/blob/master/for%20ReadMe/robustnesscheckImg.png" alt="Illustration" width="400px"/></a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;
