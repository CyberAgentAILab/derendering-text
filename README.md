## Paper: De-rendering Stylized Texts

<img src = "example/rec0.png" title = "rec" >

Wataru Shimoda<sup>1</sup>, Daichi Haraguchi<sup>2</sup>, Seiichi Uchida<sup>2</sup>, Kota Yamaguchi<sup>1</sup>  
<sup>1</sup>CyberAgent.Inc, <sup>2</sup> Kyushu University  
Accepted to ICCV2021.
[[Publication](https://openaccess.thecvf.com/content/ICCV2021/html/Shimoda_De-Rendering_Stylized_Texts_ICCV_2021_paper.html)]
[[Arxiv](https://arxiv.org/abs/2110.01890)]
[[project-page](https://cyberagentailab.github.io/derendering-text/)]

## Introduction
This repository contains the codes for ["De-rendering stylized texts"](https://arxiv.org/abs/2110.01890).
### Concept
We propose to parse rendering parameters of stylized texts utilizing a neural net.
<img src = "example/concept.jpg" title = "concept" >

### Demo
The proposed model parses rendering parameters based on famous 2d graphic engine[[Skia.org](https://skia.org/)|[python implementation](https://github.com/kyamagu/skia-python)], which has compatibility with CSS in the Web.
We can export the estimated rendering parameters and edit texts by an off-the-shelf rendering engine.

<div align = 'center'>
<img src = "example/edit0.gif" title = "edit0" height = "220" >
<img src = "example/edit1.gif" title = "edit1" height = "220" >
</div>

## Installation

### Requirements
- Python >= 3.7
- Pytorch >= 1.8.1
- torchvision >= 0.9.1

```bash
pip install -r requiements.txt
```

### Font data
- The proposed model is trained with google fonts.  
- Download google fonts and locate in `data/fonts/` as `gfonts`.  
```bash
cd data/fonts
git clone https://github.com/google/fonts.git gfonts
```

### Pre-rendered alpha maps
- The proposed model parses rendering parameters and refines them through the differentiable rendering model, which uses pre-rendered alpha maps.  
- Generate pre-rendered alpha maps.
```bash
python -m util_lib.gen_pams
```
Pre-rendered alpha maps would be generated in `data/fonts/prerendered_alpha`.

<div align = 'center'>
<img src = "example/sample.jpg" title = "inp" height = "300" >
<img src = "example/opt.gif" title = "opt" height = "300" >
</div>


## Usage

### Test
- Download the pre-trained weight from this link
([weight](https://drive.google.com/file/d/1HBcfV0nfSluCWCHGgGerx7QNJZJpOv3h/view?usp=sharing)).  
- Locate the weight file in `weights/font100_unified.pth`.  

Example usage.
```bash
python test.py --imgfile=example/sample.jpg
```
Note
- imgfile option: path of an input image
- results would be generated in `res/`

### Data generation
in progress

### Train
in progress


## Todo
- [x] Testing codes
- [ ] Codes for the text image generator
- [ ] Training codes
- [ ] Add notebooks for the guide

## Reference
```bibtex
@InProceedings{Shimoda_2021_ICCV,
    author    = {Shimoda, Wataru and Haraguchi, Daichi and Uchida, Seiichi and Yamaguchi, Kota},
    title     = {De-Rendering Stylized Texts},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {1076-1085}
}
```

## Contact
This repository is maintained by Wataru shimoda(wataru_shimoda[at]cyberagent.co.jp).
