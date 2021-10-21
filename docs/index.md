---
layout: default
---

## De-rendering Stylized Texts

![Concept](https://raw.githubusercontent.com/CyberAgentAILab/derendering-text/master/example/concept.jpg)

Wataru Shimoda<sup>1</sup>, Daichi Haraguchi<sup>2</sup>, Seiichi Uchida<sup>2</sup>, Kota Yamaguchi<sup>1</sup>  
<sup>1</sup>CyberAgent.Inc, <sup>2</sup> Kyushu University  

### Abstruct
Editing raster text is a promising but challenging task. We propose to apply text vectorization for the task of raster text editing in display media, such as posters, web pages, or advertisements. In our approach, instead of applying image transformation or generation in the raster domain, we learn a text vectorization model to parse all the rendering parameters including text, location, size, font, style, effects, and hidden background, then utilize those parameters for reconstruction and any editing task. Our text vectorization takes advantage of differentiable text rendering to accurately reproduce the input raster text in a resolution-free parametric format. We show in the experiments that our approach can successfully parse text, styling, and background information in the unified model, and produces artifact-free text editing compared to a raster baseline.

### Video
<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/R8PinaLyci0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### Overview of the proposed model
![Concept](https://raw.githubusercontent.com/CyberAgentAILab/derendering-text/master/example/model.png)

### Optimization via differentiable text rendering
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/CyberAgentAILab/derendering-text/master/example/sample.jpg" title = "inp" height = "350" >
<img src = "https://raw.githubusercontent.com/CyberAgentAILab/derendering-text/master/example/opt.gif" title = "opt" height = "350" >
</div>

### Text edit Demo
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/CyberAgentAILab/derendering-text/master/example/edit0.gif" title = "edit0" height = "400" >
<img src = "https://raw.githubusercontent.com/CyberAgentAILab/derendering-text/master/example/edit1.gif" title = "edit1" height = "400" >
</div>

### Results1
<img src = "https://raw.githubusercontent.com/CyberAgentAILab/derendering-text/master/example/rec0.png" title = "edit0">

### Results2
<img src = "https://raw.githubusercontent.com/CyberAgentAILab/derendering-text/master/example/rec1.png" title = "edit1">

### Citation

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
