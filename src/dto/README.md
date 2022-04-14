# Data class object lists
We define four types of data class objects in the following files.  
- dto_model.py
- dto_skia.py
- dto_postprocess.py
- dto_generator.py

## dto_model.py
This file includes data classes for outputs of the vectorization model. 
Most of the data classes are for managing outputs of OCR, but `TextInfo` class further contains outputs of effect information.  
- WordInstance
- BBoxInformation
- BatchWrapperBBI
- TextInfo

## dto_skia.py
This file contains data classes for text attributes.  
- FontData
- TextFormData
- ShadowParam
- FillParam
- GradParam
- StrokeParam
- EffectParams
- EffectVisibility

### FontData
This class contains information for argments of [Skia.Font](https://kyamagu.github.io/skia-python/reference/skia.Font.html#skia.Font).  
- font_size
- font_id
- font_path

### TextFormData
This class contains some options when rendering texts.  
`vertical_text_flag` is a flag of a vertical text.  
`rotate_text_flag` is a flag of a rotated text.  
`angle` is an angle of a rotated text.  
`width_scale` is a scale for the scale against an original scale.  
Note that we handle only `width_scale` for the implementation of codes.

- vertical_text_flag
- rotate_text_flag
- angle
- width_scale

### ShadowParam
We consider below attributes for drawing shadow effects.
We get [Skia.Paint object](https://kyamagu.github.io/skia-python/reference/skia.Paint.html) for drawing shadow in [`get_shadow_paint` function](https://github.com/CyberAgentAILab/derendering-text/blob/master/src/skia_lib/skia_paintor.py#L166).  
opacity
`blur` is a parameter for [BlurImageFilter function](https://kyamagu.github.io/skia-python/reference/skia.BlurImageFilter.html).  
`dilation` is a parameter for [DilateImageFilter function](https://kyamagu.github.io/skia-python/reference/skia.DilateImageFilter.html).  
`angle` and `shift` are parameters for computing offsets of shadow.  
Note that we did not handle the `dilation` effects in the paper.

The pre-defined range of `blur` parameter: `font_size`*(0~0.25).
The pre-defined range of `angle` parameter: 0~2π.
The pre-defined range of `shift` parameter: `font_size`*(0.02~0.25).
- blur
- dilation 
- angle
- shift
- offset_y
- offset_x
- color

### FillParam
We get [Skia.Paint object](https://kyamagu.github.io/skia-python/reference/skia.Paint.html) for drawing fill effect in [`get_fill_paint` function](https://github.com/CyberAgentAILab/derendering-text/blob/master/src/skia_lib/skia_paintor.py#L139).  

- color

### GradParam
We get [Skia.Paint object](https://kyamagu.github.io/skia-python/reference/skia.Paint.html) for drawing gradient in [`get_gradient_paint` function](https://github.com/CyberAgentAILab/derendering-text/blob/master/src/skia_lib/skia_paintor.py#L147).  
We render the gradient using [`GradientShader.MakeLinear` fucntion](https://kyamagu.github.io/skia-python/reference/skia.GradientShader.html).  
- points
- colors
- colorstop

### StrokeParam
We get [Skia.Paint object](https://kyamagu.github.io/skia-python/reference/skia.Paint.html) for drawing gradient in [`get_stroke_paint` function](https://github.com/CyberAgentAILab/derendering-text/blob/master/src/skia_lib/skia_paintor.py#L157).  
The pre-defined range of `border_weight` parameter: `font_size`*(0.05~0.25).
- border_weight
- color

### EffectParams
Note, we did not handle the `gradation` effect in the paper.
- shadow_param
- fill_param
- grad_param
- stroke_param

### EffectVisibility
Flags for drawing effects.
- shadow_visibility_flag
- fill_visibility_flag
- gard_visibility_flag
- stroke_visibility_flag

## dto_postprocess.py
This file contains the following data object classes for postprocessing.
- MetaDataPostprocessing
- InputData
- TextBlobParameter
- VectorData
- OptimizeParameter
- OutputData

## dto_generator.py
This file contains the following data object classes for training data generation.
`GeneratorDataInfo`
- GeneratorDataInfo
- TextGeneratorInputHandler
- RenderingData
- TrainingFormatData

