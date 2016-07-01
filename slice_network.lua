require 'nngraph'
require 'loadcaffe'

local cmd = torch.CmdLine()

cmd:option('-modelfile', 'models/VGG_ILSVRC_16_layers.caffemodel', 'Model used for perceptual losses')
cmd:option('-protofile', 'models/VGG_ILSVRC_16_layers_deploy.prototxt', 'prototxt of the perception model')
cmd:option('-style_weight', 1e0)
cmd:option('-content_weight', 1e-3)
cmd:option('-tv_weight', 0)
cmd:option('-style_layers_t', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
cmd:option('-content_layers_t', 'relu4_2')
cmd:option('-save_path_model','models/sliced_VGG_16.t7')
cmd:option('-save_path_layers','models/sliced_VGG_16_loss_layers.t7')

local function main(params)

	local content_layers = params.content_layers_t:split(",")
	local style_layers = params.style_layers_t:split(",")

	local cnn = loadcaffe.load(params.protofile, params.modelfile):float()
	local netsize = 128*256*256

	local next_content_idx, next_style_idx = 1, 1
	local netlayers = {}
	local netoutputs = {}
	local losslayers = {}
	netlayers[1] = nn.Identity()()
	-- Output TV_Loss
	print("Setting up TV layer")
	table.insert(losslayers,"tv")
	netoutputs[#netoutputs+1] = nn.MulConstant(params.tv_weight/(3*256*256))(netlayers[#netlayers])
	for i = 1, #cnn do
		if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
			local layer = cnn:get(i)
			local name = layer.name
			netlayers[#netlayers+1] = layer(netlayers[#netlayers])
			if name == content_layers[next_content_idx] then
				-- Output content losses
				print("Setting up content layer", i, ":", layer.name)
				table.insert(losslayers,"content")
				netoutputs[#netoutputs+1] = nn.MulConstant(params.content_weight/(netsize/2^32))(netlayers[#netlayers])
				next_content_idx = next_content_idx + 1
			end
			if name == style_layers[next_style_idx] then
				-- Output style losses
				print("Setting up style layer  ", i, ":", layer.name)
				table.insert(losslayers,"style")
				netoutputs[#netoutputs+1] = nn.MulConstant(params.style_weight/(netsize/2^next_style_idx))(GramMatrix()(netlayers[#netlayers]))
				next_style_idx = next_style_idx + 1
			end
		end
	end
	local preparedModel = nn.gModule({netlayers[1]},netoutputs)
	print("Saving sliced model...")
	torch.save(params.save_path_model,preparedModel:float())
	print("Saving loss layers...")
	torch.save(params.save_path_layers,losslayers)
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

local params = cmd:parse(arg)
main(params)