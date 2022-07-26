from dataclasses import dataclass
from typing import Optional, List


@dataclass
class FontData:
    font_size: float
    font_id: int
    font_path: str

    def get_data(self):
        return self.font_size, self.font_id, self.font_path


@dataclass
class TextFormData:
    vertical_text_flag: bool
    rotate_text_flag: bool
    angle: float
    width_scale: float
    text_index: int

    def get_data(self):
        return self.vertical_text_flag, self.rotate_text_flag, self.angle, self.width_scale


@dataclass
class ShadowParam:
    opacity: float
    blur: float
    dilation: float
    angle: float
    shift: float
    offset_y: float
    offset_x: float
    color: List[float]

    def get_data(self):
        return self.opacity, self.blur, self.dilation, self.angle, self.shift, self.offset_y, self.offset_x, self.color


@dataclass
class FillParam:
    color: List[float]

    def get_data(self):
        return self.color


@dataclass
class GradParam:
    grad_mode: int
    blend_mode: int
    points: List
    colors: List
    colorstop: List

    def get_data(self):
        return self.grad_mode, self.blend_mode, self.points, self.colors, self.colorstop


@dataclass
class StrokeParam:
    border_weight: float
    color: List[float]

    def get_data(self):
        return self.border_weight, self.color


@dataclass
class EffectParams:
    shadow_param: ShadowParam
    fill_param: FillParam
    grad_param: Optional[GradParam]
    stroke_param: StrokeParam

    def get_shadow_param(self):
        return self.shadow_param.get_data()

    def get_fill_param(self):
        return self.fill_param.get_data()

    def get_grad_param(self):
        if self.grad_param is None:
            return None
        else:
            return self.grad_param.get_data()

    def get_stroke_param(self):
        return self.stroke_param.get_data()

    def get_data(self):
        return self.get_shadow_param(), self.get_fill_param(
        ), self.get_grad_param(), self.get_stroke_param()


@dataclass
class EffectVisibility:
    shadow_visibility_flag: bool
    fill_visibility_flag: bool
    gard_visibility_flag: bool
    stroke_visibility_flag: bool

    def get_data(self):
        return self.shadow_visibility_flag, self.fill_visibility_flag, self.gard_visibility_flag, self.stroke_visibility_flag
