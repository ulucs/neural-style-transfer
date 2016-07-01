require 'imgtools'
require 'nngraph'
require 'optim'

local cmd = torch.CmdLine()

cmd:option('-style_img', 'style_images/starry.jpg', 'Style image to train on the image set')
cmd:option('-content_img', 'image_sets/tubingen.jpg', 'Image to apply the style on')
cmd:option('-output','output.jpg','Path to output')
cmd:option('-iterations', 100, 'Number of iterations to run for training')
cmd:option('-learning_rate',5)
cmd:option('-model_path','models/sliced_VGG_16.t7','Location of the pre-sliced model')
cmd:option('-loss_layers_path', 'models/sliced_VGG_16_loss_layers.t7', 'Loss layers of the pre-sliced model')

local function main(params)

	-- Load the content and style images
	local content_img = loadimg(params.content_img,256):float()
	local content_batch_size = torch.LongStorage(4)
	local orgSize = #content_img
	content_batch_size[1] = 1
	for i=1,3 do
		content_batch_size[i+1] = orgSize[i]
	end
	content_img = torch.reshape(content_img,content_batch_size)

	local style_img = loadimg(params.style_img,256):float()
	local style_batch_size = torch.LongStorage(4)
	style_batch_size[1] = 1
	for i=1,3 do
		style_batch_size[i+1] = (#style_img)[i]
	end
	style_img = torch.reshape(style_img,style_batch_size)
	
	-- Load up the networks
	local perception_model = torch.load(params.model_path)
	local losslayers = torch.load(params.loss_layers_path)


	-- Our model will transfer tv and content loss from content image, so we replace those
	-- y represents the values we're trying to optimize towards
	local yc = perception_model:forward(content_img)
	local yg = {}

	for i = 1, #losslayers do
		local lossname = losslayers[i]
		if lossname == "tv" or lossname == "content" then
			yg[i] = yc[i]:clone()
		end
	end

	local y = perception_model:forward(style_img)
	for i = 1, #losslayers do
		local lossname = losslayers[i]
		if lossname == "style" then
			yg[i] = y[i]:clone()
		end
	end
	y = nil
	yc = nil

	-- Our criterion is MSE, we set it and the gradients up here
	local criterion = nn.ParallelCriterion()
	for i = 1, #losslayers do
		criterion:add(nn.MSECriterion())
	end
	criterion:float()
	local _, gradParams = perception_model:getParameters()

	-- Define the loss&gradient function, optimizer's state
	local function feval(x)
		gradParams:zero()

		-- Just run the network back and forth
		local yhat = perception_model:forward(x)
		local loss = criterion:forward(yhat,yg)
		local loss_grads = criterion:backward(yhat,yg)
		local grads = perception_model:backward(x,loss_grads)

		collectgarbage()
		return loss, grads:view(grads:nElement())
	end

	local optim_state = {learningRate = params.learning_rate}

	-- Run the optimization for n iterations
	print('Running optimization with ADAM')
	for t = 1, params.iterations do
		local x, losses = optim.adam(feval, content_img, optim_state)
		print('Iteration number: '.. t ..'; Current loss: '.. losses[1])
	end

	-- Save the optimized image
	content_img = torch.reshape(content_img,orgSize)
	saveimg(content_img,params.output)

end

local params = cmd:parse(arg)
main(params)