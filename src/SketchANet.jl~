include("Sketch.jl")
include("Preprocessor.jl")
include("SketchGetter.jl")

module SketchANet
using Knet
using Preprocessor
using PyPlot
global batchsize = 50
global datapath = "/mnt/kufs/scratch/kkaiyrbekov15/Sketch Network/data"


function loss(w, x, ygold)
  ypred = predict(w,x)
  ynorm = logp(ypred,1)
  return -sum(ygold .* ynorm)/size(ygold, 2)
end

function countcorrect(ypred, ygold)
	correct = sum(ygold .* (ypred .== maximum(ypred,1)))
	return correct
end


function accuracy(w, dataind; atype = Array{Float32} )
  batchcount = Integer(length(dataind)/batchsize)
  correct = 0.0
  softloss = 0.0
  for batchnum=1:batchcount
    start = (batchnum-1)*batchsize + 1
    finish = min(start + batchsize - 1, length(dataind))
    batchind = dataind[start:finish]
    (x,ygold) = getbatch(batchind; atype=atype)
    ypred = predict(w, x)
    correct += countcorrect(ypred, ygold)
#    println("batch: $batchnum correct: $correct")
    softloss += loss(w, x, ygold)
  end
  return correct/length(dataind), softloss/length(dataind)
end

lossgradient = grad(loss)

function train(w, data; lr = 0.1, epochs = 20, atype = Array{Float32})
  for epoch=1:epochs
    for (x,y) in data
      x = convert(atype, x)
      y = convert(atype, y)
      g = lossgradient(w, x, y)
      for i=1:length(w)
        axpy!(-lr, g[i], w[i])
      end
    end
  end
  return w
end

function trainvanilla(w, prms, data; lr = 0.1, epochs = 1, atype = Array{Float32})
  #check for edge cases of bacthcount
  for epoch=1:epochs
    for (x,y) in data
      newx = perturb(x)
      newx = convert(atype, newx)
      y = convert(atype, y)
      g = lossgradient(w, newx, y)
      for i=1:length(w)
        axpy!(-lr, g[i], w[i])
      end
    end
  end
  return w
end

function perturb( x )
  new_x = zeros(Float32, size(x, 1), size(x, 2), size(x, 3), batchsize)
  for i=1:batchsize
    roll = rand()
    if roll > 0.45
      new_x[:, :, :, i] = imrotate(x[:,:,:,i], rand(-5:5))
    else
      new_x[:, :, :, i] = x[:, :, :, i]
    end
    roll = rand()
    if roll > 0.5
      new_x[:, :, :, i] = flipdim(new_x[:, :, :, i], 2)
    end
  end
  return new_x
end

function accuracyvanilla(w, data; atype = Array{Float32} )
  ninstance = 0
  softloss = 0
  correct = 0
  for (x,ygold) in data
    x = convert(atype, x)
    ygold = convert(atype, ygold)
    ypred = predict(w, x; dprob=0)
    correct += countcorrect(ypred, ygold)
#    println("batch: $batchnum correct: $correct")
    softloss += loss(w, x, ygold)
    ninstance += size(ygold, 2)
  end
  return correct/ninstance, softloss/ninstance
end

function dropout(x, prob)
  #individual nodes are dropped out of net with probability=1-prob and kept with probability=prob
  if prob > 0
     (x .* convert(KnetArray{Float32}, (rand!( zeros( size(x) ) )).> prob )) ./ (1-prob)
  else
     x
  end
end

function predict(w, x; dprob=0.5)
  x = pool(relu(conv4(w[1],x;padding=0, stride=3) .+ w[2]); window=3, stride=2)
  x = pool(relu(conv4(w[3],x;padding=0, stride=1) .+ w[4]); window=3, stride=2)
  for i=5:2:9
    x = relu(conv4(w[i],x;padding=(1,1), stride=1) .+ w[i+1])
  end
  x = pool(x; window=3, stride = 2)
  #fully connected layers
  x = mat(x)
  x = relu(w[11]*x .+ w[12])
  x = dropout(x, dprob)
  x = relu(w[13]*x .+ w[14])
  x = dropout(x, dprob)

  return w[end-1]*x .+ w[end]
end

function initdata(dataind; train = false)
  data = Any[]
  cnt = 5
  trmcount = train?cnt:1
  for i=1:trmcount
    minibatch!(data, dataind, i)
  end
  return data
end

function minibatch!(data, dataind, imgindx)
  #=
  data -> array of (x,y) batches to update
  dataind -> array of all file indices range = [1,20000]
  imgindx -> index of an augmented image range = [1, # of tranformations]
  =#
  numbatches = Integer(length(dataind)/batchsize)
  for batchnum=1:numbatches
    start = (batchnum-1)*batchsize + 1
    finish = min(start + batchsize - 1, length(dataind))
    push!(data, getbatch(view(dataind, start:finish), imgindx))
  end
end

function getbatch(batchind, imgindx; classnum = 250, channels = 1)
  completeness = 100
  imsize = 225
  len = length(batchind)
  #define data and labels for batch
  data = Array{Float32}(imsize, imsize, channels, len)
  labels = zeros(Float32, classnum, len)
  #initialize each instance and label of batch
  for i=1:len
    filenum = batchind[i]
    class, instancenum = filenum2ind(filenum)
    labels[class, i] = 1
    path = "$datapath/$(class)/$(completeness)/$(instancenum)/$(imgindx).png"
    data[:, :, :, i] = getSketchMatrix(path; imsize=imsize)
  end
  return (data, labels)
end

function initparams(weights)
  prms = Any[]
  for i=1:length(weights)
    push!(prms, Adam(weights[i]; lr = 0.1))
  end
  return prms
end


function xavier(a...)
  w = rand(a...)
  # The old implementation was not right for fully connected layers:
  # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
  if ndims(w) < 2
    error("ndims=$(ndims(w)) in xavier")
  elseif ndims(w) == 2
    fanout = size(w,1)
    fanin = size(w,2)
  else
    fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
    fanin = div(length(w), fanout)
  end
  # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  s = sqrt(2 / (fanin + fanout))
  w = 2s*w-s
end

function weights(;atype=KnetArray{Float32}, channels = 1, winit = 0.001)
  w = Array(Any, 16)
  #Convolutional layers
  #Layer No 1
  w[1] = xavier(15, 15, channels, 64)
  w[2] = zeros(1, 1, 64, 1)
  #Layer No 2
  w[3] = xavier(5, 5, 64, 128)
  w[4] = zeros(1, 1, 128, 1)
  #Layer No 3
  w[5] = xavier(3, 3, 128, 256)
  w[6] = zeros(1, 1, 256, 1)
  #Layer No 4
  w[7] = xavier(3, 3, 256, 256)
  w[8] = zeros(1, 1, 256, 1)
  #Layer No 5
  w[9] = xavier(3, 3, 256, 256)
  w[10] = zeros(1, 1, 256, 1)

  #Fully connected layers
  #Layer 6
  w[11] = xavier(512, 12544)
  w[12] = zeros(512, 1)
  #Layer 7
  w[13] = xavier(512, 512)
  w[14] = zeros(512, 1)
  #Layer 8
  w[15] = xavier(250, 512)
  w[16] = zeros(250, 1)
  return map(a->convert(atype, a), w)
end

function filenum2ind(filenum::Integer; N = 80)
  class = Integer(floor((filenum-1)/N) + 1)
  instancenum = filenum - (class-1)*N
  return class, instancenum
end

function gettraintestindices(dataperclass; trainpercent=0.67, classnum = 250, N = 80)
  #=
  -classnum = 250 -> number of classes
  -N = 80 -> number of instances per class
  -trainpercent -> percentage of instances per class to use for training
  =#
  #number of training samples per class
  traincount = Integer(round(trainpercent*dataperclass))
  #random permutation of indexes in single class
  perm = randperm(dataperclass)
  trnind = perm[1:traincount] #file indices for training
  tstind = perm[traincount+1:end] #file indices for testing
  trncpy = copy(trnind)
  tstcpy = copy(tstind)
  #add corresponding entries of each class
  for c=2:classnum
    trnind = vcat(trnind, (c-1)*N .+ trncpy)
    tstind = vcat(tstind, (c-1)*N .+ tstcpy)
  end
  return shuffle(trnind), shuffle(tstind)
end

function main()
  dataperclass = 80
  #global batchsize = 50
  #global datapath = "/mnt/kufs/scratch/kkaiyrbekov15/Sketch Network/data"
  lr = 0.01
  trnind, tstind = gettraintestindices(dataperclass)
  if gpu() >= 0
    atype = KnetArray{Float32}
  else
    atype = Array{Float32}
  end
  w = weights(atype=atype)
  prms = initparams(w)

  println("Type of weights array $(typeof(w))")
  datatrn = initdata(trnind; train = true)
  datatst = initdata(tstind; train = false)
  println("Data minibatched successfully")

  trnlosses = []
  tstlosses = []
  trnerr  = []
  tsterr  = []
  iters = []
  for i=1:100
    w = trainvanilla(w, prms, datatrn; lr=lr, epochs = 3, atype=atype)
    trnacc, trnloss = accuracyvanilla(w, datatrn; atype=atype)
    tstacc, tstloss = accuracyvanilla(w, datatst; atype=atype)
    if lr%10 == 0
      lr = lr/2
    end
    append!(trnlosses, trnloss)
		append!(tstlosses, tstloss)
		append!(iters, i)
		append!(tsterr, 1-tstacc)
		append!(trnerr, 1-trnacc)
    @printf("epoch: %d trn accuracy-loss: %g - %g tst accuracy-loss: %g - %g\n", i, trnacc, trnloss, tstacc, tstloss)
  end

  figure()
	title("Loss functions")
	plot(iters, trnlosses, label = "train loss")
	plot(iters, tstlosses, label = "test loss")
	xlabel("epochs")
	ylabel("loss")
	legend()
	savefig("lossaug5.png")
	close()
	figure()
	title("Errors")
	plot(iters, trnerr, label = "train error")
	plot(iters, tsterr, label = "test error")
	xlabel("epochs")
	ylabel("error")
	legend()
	savefig("erraug5.png")
	close()

#=
  for i=1:50
  	w = train(w, trnind; lr = 0.01, epochs = 4,  atype=atype)
  	trnacc, trnloss = accuracy(w, trnind;  atype=atype)
  	tstacc, tstloss = accuracy(w, tstind;  atype=atype)
  	@printf("epoch: %d trn accuracy-loss: %g - %g tst accuracy-loss: %g - %g\n", i, trnacc, trnloss, tstacc, tstloss)
  end
  =#
end

main()
end
