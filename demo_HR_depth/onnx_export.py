import torch
import torch.onnx
import networks

# pretrained model parameter files:
depth_encoder_path = "./models/HR_Depth_K_M_1280x384/encoder.pth"
depth_decoder_path = "./models/HR_Depth_K_M_1280x384/depth.pth"

# convert the encoder to ONNX
depth_encoder = networks.ResnetEncoder(18, False)

encoder_dict = torch.load(depth_encoder_path, map_location=torch.device('cpu'))
img_height = encoder_dict["height"]
img_width = encoder_dict["width"]
print("Test image height is:", img_height)
print("Test image width is:", img_width)

load_dict = {k: v for k, v in encoder_dict.items() if k in depth_encoder.state_dict()}
depth_encoder.load_state_dict(load_dict)
depth_encoder.eval()

dummy_input_a = torch.randn(1, 3, img_height, img_width)

torch.onnx.export(depth_encoder,
    dummy_input_a,
    "./depth_encoder.onnx",
    export_params=True,
    #opset_version=10,
    do_constant_folding=True,
    input_names=["input.0"],
    output_names=["output.0", "output.1", "output.2", "output.3", "output.4"],
    #dynamic_axes={}
    )

# convert decoder to ONNX
depth_decoder = networks.HRDepthDecoder(depth_encoder.num_ch_enc)

decoder_dict = torch.load(depth_decoder_path, map_location=torch.device('cpu'))
depth_decoder.load_state_dict(decoder_dict)
depth_decoder.eval()

dummy_input0 = torch.randn(1,  64, 192, 640)
dummy_input1 = torch.randn(1,  64,  96, 320)
dummy_input2 = torch.randn(1, 128,  48, 160)
dummy_input3 = torch.randn(1, 256,  24,  80)
dummy_input4 = torch.randn(1, 512,  12,  40)
dummy_input_b  = [dummy_input0,dummy_input1,dummy_input2,dummy_input3,dummy_input4]

torch.onnx.export(depth_decoder,
    dummy_input_b,
    "./depth_decoder.onnx",
    export_params=True,
    #opset_version=10,
    do_constant_folding=True,
    input_names=["input.0", "input.1", "input.2", "input.3", "input.4"],
    output_names=["scale0", "scale1", "scale2", "sacle3"],
    #dynamic_axes={}
    )
