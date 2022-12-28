import onnx

model = onnx.load_model("centerface.onnx")
d = model.graph.input[0].type.tensor_type.shape.dim
    d[0].dim_value = 1
d[2].dim_value = -1
d[3].dim_value = -1
onnx.save_model(model,"centerface_dynamic.onnx" )
