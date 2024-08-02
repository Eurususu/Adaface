#coding=utf-8
import torch
import torch.nn as nn
import net


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_webface4m.ckpt",
}
def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


if __name__ == '__main__':
    model = load_pretrained_model('ir_50')
    input_names = ['input']
    output_names = ['output0', 'output1']
    x = torch.randn(1, 3, 112, 112)
    # 这里的0表示第一个维度为动态，batch表示对这个维度起的名字
    dynamic_axes_0 = {
        'input': {0: 'batch'},
        'output0': {0: 'batch'},
        'output1': {0: 'batch'}
    }
    torch.onnx.export(model, x, 'adaface_r50_web4m.onnx', input_names=input_names, output_names=output_names,
                      verbose=False, opset_version=13,
                      do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
                      dynamic_axes=dynamic_axes_0)





