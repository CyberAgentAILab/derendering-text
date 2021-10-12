## Text Image Generator
This work is inspired by [SynthText](https://github.com/ankush-me/SynthText).
We focus on the text images for displayed media such as posters, web pages, or advertisements, then the text image generator generates simple but stylized texts.

## Introduction
This text image generator repeats following steps and generates text images.
- Sample parameters for texts
- Render texts for the parameters

### Renderer
This text image generator is based on [skia-python](https://github.com/kyamagu/skia-python), which is a modern 2d graphic engine.

#### Effects
Currently, we support following effects.
- Shadow
- Fill
- Border
- Gradation(not used in the training of the model in the paper)


### Sampler
Our sampler consists of three steps as following.
##### Text and font sampler `src/sampler/text_font_sampler.py`
- Text contents
- Font
##### Offset sampler `src/sampler/offset_sampler.py`
- Text spatial information
##### Style sampler `src/sampler/style_sampler.py`
- Effect information

## Usage

Quick start.
```bash
python gen.py --bgtype=load --bg_dir=src/modules/generator/example/bg --mask_dir=src/modules/generator/example/mask
```

## Configurations

### Dataset
#### SynthText dataset

#### FMD dataset
Download datasets from [link](https://people.csail.mit.edu/celiu/CVPR2010/FMD/)
#### BAM dataset

#### Book dataset
#### Own dataset

