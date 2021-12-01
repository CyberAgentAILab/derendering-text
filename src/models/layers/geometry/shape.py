def convert_shape(outs, batch_num, text_num, training, text_pool_num):
    converted_outs = []
    for i in range(len(outs)):
        if training == True:
            out = outs[i].view(
                batch_num,
                text_pool_num,
                outs[i].shape[1],
                outs[i].shape[2],
                outs[i].shape[3])
        else:
            out = outs[i].view(
                1,
                text_num,
                outs[i].shape[1],
                outs[i].shape[2],
                outs[i].shape[3])
        converted_outs.append(out)
    return converted_outs
