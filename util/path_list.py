import os


def get_weight():
    return "weights/font_100_unified_model.pth"


def get_char_dict_file_path():
    return "data/char_dict/eng_94.txt"


def get_load_data_path():
    return "data"


def get_generator_load_data_path():
    return "data"


def get_generator_save_data_path():
    return "gen_data"


def get_prerendered_alpha_dir():
    return "data/fonts/prerendered_alpha"


def get_google_font_path():
    return os.path.join(get_load_data_path(), "fonts/gfonts")


def get_google_font_list_filename():
    return os.path.join(
        get_load_data_path(),
        "fonts/font_list/latin_ofl_100.txt")


def get_newsgroup_text_courpas():
    return os.path.join(
        get_generator_load_data_path(),
        "newsgroup/newsgroup.txt")


def get_fmd_data_dir():
    return os.path.join(
        get_generator_load_data_path(),
        "fmd")
