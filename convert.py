import torch
import torch.nn as nn
import numpy as np
import caffe
from ssrnet import MTSSRNet

if __name__ == "__main__":
    # model = MTSSRNet()
    # model.eval()
    # torch.save(model.state_dict(), "pred.pth")
    model = torch.load("./pred_15.pth", map_location = "cpu")
    net = caffe.Net("ssr_pred.prototxt", caffe.TEST)
    to_save_keys = net.params.keys()
    ori_keys = model.keys()

    for key_and_type in to_save_keys:
        key = "_".join(key_and_type.split("_")[:-1])
        tp = key_and_type.split("_")[-1]

        if tp == "conv":
            # weight
            print(key_and_type)
            #if key_and_type.split(".")[0].endswith("t"):
            #    net.params[key_and_type][0].data[:] = model[key.replace("t.", "_.") + ".weight"].cpu().numpy()
            #    # bias
            #    if (key + ".bias") in ori_keys:
            #        net.params[key_and_type][1].data[:] = model[key.replace("t.", "_.") + ".bias"].cpu().numpy()
            #    continue
            net.params[key_and_type][0].data[:] = model[key + ".weight"].cpu().numpy()
            # bias
            if (key + ".bias") in ori_keys:
                net.params[key_and_type][1].data[:] = model[key + ".bias"].cpu().numpy()
        elif tp == "bn":
            print(key_and_type)
            # weight
            net.params[key_and_type][0].data[:] = model[key + ".running_mean"].cpu().numpy()
            net.params[key_and_type][1].data[:] = model[key + ".running_var"].cpu().numpy()
            net.params[key_and_type][2].data[:] = 1
        elif tp == "scale":
            print(key_and_type)
            net.params[key_and_type][0].data[:] = model[key + ".weight"].cpu().numpy()
            net.params[key_and_type][1].data[:] = model[key + ".bias"].cpu().numpy()
        elif tp == "innerproduct":
            print(key_and_type)
            # weight
            net.params[key_and_type][0].data[:] = model[key + ".weight"].cpu().numpy()
            # bias
            if (key + ".bias") in ori_keys:
                net.params[key_and_type][1].data[:] = model[key + ".bias"].cpu().numpy()
        else:
            print(key_and_type , "no", "*" * 10)
    net.save("./ssr.caffemodel")

    

    
