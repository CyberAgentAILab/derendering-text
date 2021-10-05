def convert_shape(outs, batch_num, text_num):
    converted_outs = []
    for i in range(len(outs)):
        out = outs[i].view(
            batch_num,
            text_num,
            outs[i].shape[1],
            outs[i].shape[2],
            outs[i].shape[3])
        converted_outs.append(out)
    return converted_outs
