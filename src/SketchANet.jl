include("Sketch.jl")
include("Preprocessor.jl")
include("SketchGetter.jl")

module SketchANet
using Knet
using Preprocessor
using PyPlot
using JLD, ArgParse
global const batchsize = 50
global const datapath = "/mnt/kufs/scratch/kkaiyrbekov15/Sketch Network/data"
global const jldpath = "/mnt/kufs/scratch/kkaiyrbekov15/Sketch Network/dataJLD"


function loss(w, x, ygold; lambda=0.000001)
  ypred = predict(w,x)
  ynorm = logp(ypred,1)
#= reg = 0
  if lambda != 0
    reg = (lambda/2) * sum(sumabs2(wi) for wi in w[1:2:end])
  end=#
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

function train(w, prms, data, meansketch; limit = length(data), lr = 0.1, epochs = 1, atype = Array{Float32})
  for epoch=1:epochs
    for (x,y) in data
      newx = perturb(x, meansketch)
      newx = convert(atype, newx)
      y = convert(atype, y)
      g = lossgradient(w, newx, y)
      for i=1:length(w)
        update!(w[i], g[i], prms[i])
      end
      if limit < 0
        break
      end
      limit -= 1
    end
  end
  return w
end

function trainvanilla(w, prms, data, meansketch; limit = length(data), lr = 0.1, epochs = 1, atype = Array{Float32})
  #check for edge cases of bacthcount
  for epoch=1:epochs
    for (x,y) in data
      newx = perturb(x, meansketch)
      newx = convert(atype, newx)
      y = convert(atype, y)
      g = lossgradient(w, newx, y)
      for i=1:length(w)
        axpy!(-lr, g[i], w[i])
      end
      if limit < 0
        break
      end
      limit -= 1
    end
  end
  return w
end

function perturb( x, meansketch)
  new_x = zeros(Float32, size(x, 1), size(x, 2), size(x, 3), batchsize)
  intvl = -32:32
  scales = 0.7:0.1:1.3
  for i=1:size(x, 4)
    roll = rand()
    shiftr = rand(intvl)
    shiftc = rand(intvl)
    scale =  rand(scales)
    if roll > 0.45
      new_x[:, :, :, i] = imscale(x[:, :, :, i], new_x[:, :, :, i], scale)
    end
    roll = rand()
    if roll > 0.45
      new_x[:, :, :, i] = imrotate(x[:,:,:,i], rand(-33:3:33))
    else
      new_x[:, :, :, i] = x[:, :, :, i]
    end
    new_x[:, :, :, i] = circshift(new_x[:, :, :, i], [shiftr shiftc 0])
    roll = rand()
    if roll > 0.5
      new_x[:, :, :, i] = flipdim(new_x[:, :, :, i], 2)
    end
    new_x[:, :, :, i] -= meansketch
  end
  return new_x
end

function accuracyvanilla(w, data, meansketch; limit = length(data), atype = Array{Float32} )
  ninstance = 0
  softloss = 0
  correct = 0
  for (x,ygold) in data
    x = x .- meansketch
    x = convert(atype, x)
    ygold = convert(atype, ygold)
    ypred = predict(w, x; dprob=0)
    correct += countcorrect(ypred, ygold)
#    println("batch: $batchnum correct: $correct")
    softloss += loss(w, x, ygold)
    ninstance += size(ygold, 2)
    if limit < 0
      break
    end
    limit -= 1
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

function initdata(dataind; train::Bool = false, imsize::Int = 225)
  data = Any[]
  cnt = 1
  trmcount = train?cnt:1
  for i=1:trmcount
    minibatch!(data, dataind, i; imsize=imsize)
  end
  return data
end

function minibatch!(data, dataind, imgindx; imsize::Int = 225)
  #=
  data -> array of (x,y) batches to update
  dataind -> array of all file indices range = [1,20000]
  imgindx -> index of an augmented image range = [1, # of tranformations]
  =#
  numbatches = Integer(length(dataind)/batchsize)
  for batchnum=1:numbatches
    start = (batchnum-1)*batchsize + 1
    finish = min(start + batchsize - 1, length(dataind))
    push!(data, getbatch(view(dataind, start:finish), imgindx; imsize=imsize))
  end
end

function getbatch(batchind, imindex; classnum::Int = 250, imsize::Int = 225, trgsize::Int = 225)
  len = length(batchind)
  #define data and labels for batch
  data = Array{Float32}(trgsize, trgsize, channels, len)
  labels = zeros(Float32, classnum, len)
  #initialize each instance and label of batch
  fillbatch!(data, labels, imindex, imsize, batchind, len)
  return (data, labels)
end

function fillbatch!(data, labels, imindex, imsize::Int, batchind, len)
  for i=1:len
    filenum = batchind[i]
    class, instancenum = filenum2ind(filenum)
    labels[class, i] = 1
    initvolume!(data, i, datapath, class, cmap, instancenum, imindex; imsize = imsize)
    #println(sum(data[:, :, :, i]))
  end
end

function initparams(weights)
  prms = map(x->Sgd(lr=0.01,gclip=0), weights)
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

function weights(;atype=KnetArray{Float32}, bias=0.01, winit = 0.01)
  w = Array(Any, 16)
  #Convolutional layers
  #Layer No 1
  w[1] = xavier(15, 15, channels, 64)
  w[2] = bias*ones(1, 1, 64, 1)
  #Layer No 2
  w[3] = xavier(5, 5, 64, 128)
  w[4] = bias*ones(1, 1, 128, 1)
  #Layer No 3
  w[5] = xavier(3, 3, 128, 256)
  w[6] = bias*ones(1, 1, 256, 1)
  #Layer No 4
  w[7] = xavier(3, 3, 256, 256)
  w[8] = bias*ones(1, 1, 256, 1)
  #Layer No 5
  w[9] = xavier(3, 3, 256, 256)
  w[10] = bias*ones(1, 1, 256, 1)

  #Fully connected layers
  #Layer 6
  w[11] = xavier(512, 12544)
  w[12] = bias*ones(512, 1)
  #Layer 7
  w[13] = xavier(512, 512)
  w[14] = bias*ones(512, 1)
  #Layer 8
  w[15] = xavier(250, 512)
  w[16] = bias*ones(250, 1)
  return map(a->convert(atype, a), w)
end

function filenum2ind(filenum::Int; N::Int = 80)
  class = Integer(floor((filenum-1)/N) + 1)
  instancenum = filenum - (class-1)*N
  return class, instancenum
end

function gettraintestindices(dataperclass; trainpercent::Float64 = 0.67, classnum::Int = 250, N::Int = 80, readyindx::Bool = false)
  #=
  -classnum = 250 -> number of classes
  -N = 80 -> number of instances per class
  -trainpercent -> percentage of instances per class to use for training
  =#
  #if data partition is ready then load it
  if readyindx
    println("loading indices")
    trnind = load("$(jldpath)/trainindices.jld")["trnind"]
    tstind = load("$(jldpath)/testindices.jld")["tstind"]
    return trnind, tstind
  end
  #number of training samples per classs
  traincount = Int(round(trainpercent*dataperclass))
  first  = Int(round(traincount/2))
  second = traincount - first
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
  trnind = shuffle(trnind)
  tstind = shuffle(tstind)
#  save("$(jldpath)/trainindices.jld","trnind", trnind)
#  save("$(jldpath)/testindices.jld","tstind", tstind)
  return trnind, tstind
end

function initchannelmap()
  cmap = Array{Int}(6)
  cmap[1] = 100
  cmap[2] = 80
  cmap[3] = 60
  cmap[4] = 40
  cmap[5] = 20
  cmap[6] = 70
  return cmap
end


function split(dataperclass; trainpercent::Float64 = 0.67, classnum::Int = 250, N::Int = 80, readyindx::Bool = false)
  #number of training samples per classs
  traincount = Int(round(trainpercent*dataperclass))
  first  = Int(round(traincount/2))
  second = traincount - first
  #random permutation of indexes in single class
  perm = randperm(dataperclass)
  #file indices for splits 1,2,3
  part1 = perm[1:first]
  part2 = perm[first+1:second]
  part3 = perm[second+1:end]
  part1cpy = copy(part1)
  part2cpy = copy(part2)
  part3cpy = copy(part3)
  #add corresponding entries of each class
  for c=2:classnum
    part1 = vcat(part1, (c-1)*N .+ part1)
    part2 = vcat(part2, (c-1)*N .+ part2)
    part3 = vcat(part3, (c-1)*N .+ part3)
  end
  part1 = shuffle(part1)
  part2 = shuffle(part2)
  part3 = shuffle(part3)
  return part1, part2, part3
end

function saveplots(iters, trnlosses, tstlosses, trnerr, tsterr, lossfigname::String, errfigname::String)
  figure()
  title("Loss functions")
  plot(iters, trnlosses, label = "train loss")
  plot(iters, tstlosses, label = "test loss")
  xlabel("epochs")
  ylabel("loss")
  legend()
  savefig(lossfigname)
  close()
  figure()
  title("Errors")
  plot(iters, trnerr, label = "train error")
  plot(iters, tsterr, label = "test error")
  xlabel("epochs")
  ylabel("error")
  legend()
  savefig(errfigname)
  close()
end

# To be able to load/save KnetArrays:
if Pkg.installed("JLD") != nothing
  import JLD: writeas, readas
  type KnetJLD; a::Array; end
  writeas(c::KnetArray) = KnetJLD(Array(c))
  readas(d::KnetJLD) = KnetArray(d.a)
end

function test(data, meansketch; scales = [225 256 128 192 64], atype = KnetArray{Float32})
  disp = 32
  shifts = [0 0; disp disp; -disp disp; disp -disp; -disp -disp]
  models = Any[]
  #load models for different scales
  for i in length(scales)
    w = load("$(jldpath)/Models/FloatModel/zero/floatmodel$(scales[i]).jld")["model"]
    w = map(a->convert(atype, a), w)
    push!(models, w)
  end
  info("Models are loaded!")
  #initialize array for predictions
  ypred = Any[]
  for bind=1:length(data)
    push!(ypred, KnetArray(zeros(Float32, size(data[bind][2]))))
  end
  info("Prediction arrays have been initialized!")
  #accumulate predictions for each model
  for i in length(models)
    #update predictions using current model
    model = models[i]
    updateypred!(data, model, ypred, shifts, meansketch; atype=atype)
  end
  info("Fused predictions obtained!")
  correct = 0.0
  ninstance = 0.0
  for bind=1:length(data)
    correct += countcorrect(ypred[bind], convert(atype, data[bind][2]))
    ninstance += size(ypred[bind], 2)
  end
  return correct/ninstance
end

function updateypred!(data, model, ypred, shifts, meansketch; atype = KnetArray{Float32})
  #iteratte over data
  for bind=1:length(data)
    x = data[bind][1]
    shiftcnt = size(shifts, 1)
    yptmp = KnetArray(zeros(Float32, size(ypred[bind])))
    #iterate over transformations
    for j=1:shiftcnt
      shiftr = shifts[j , 1]
      shiftc = shifts[j , 2]
      xshifted = circshift(x, [shiftr shiftc 0 0]) .- meansketch
      xflipped = flipdim(xshifted, 2) .- meansketch
      xshifted = convert(atype, xshifted)
      yptmp = yptmp .+ logp(predict(model, xshifted; dprob=0), 1)
      xflipped = convert(atype, xflipped)
      yptmp  = yptmp .+ logp(predict(model, xflipped; dprob=0), 1)
    end
    ypred[bind] = ypred[bind] .+ (yptmp ./ shiftcnt)
  end
end


function extractfeat(w, x)
  #Extracts features of the penultimate layer
  x = pool(relu(conv4(w[1],x;padding=0, stride=3) .+ w[2]); window=3, stride=2)
  x = pool(relu(conv4(w[3],x;padding=0, stride=1) .+ w[4]); window=3, stride=2)
  for i=5:2:9
    x = relu(conv4(w[i],x;padding=(1,1), stride=1) .+ w[i+1])
  end
  x = pool(x; window=3, stride = 2)
  #fully connected layers
  x = mat(x)
  x = relu(w[11]*x .+ w[12])
  return relu(w[13]*x .+ w[14])
end

function getfeats(w, data; atype = KnetArray{Float32})
  features = nothing
  for (x,y) in data
    x = convert(atype, x)
    feat = extractfeat(w, x)
    if features == nothing
      features = feat
    else
      features = hcat(features, feat)
    end
  end
  return features
end

function savefeats(data; scales = [225 256 128 192 64], atype = KnetArray{Float32}, train::Bool = false)
  models = Any[]
  #load models for different scales
  for i in length(scales)
    w = load("$(jldpath)/Models/FloatModel/zero/floatmodel$(scales[i]).jld")["model"]
    w = map(a->convert(atype, a), w)
    push!(models, w)
  end
  info("Models are loaded!")
  features = []
  for i in length(models)
    #get and concatenate features
    model = models[i]
    if i == 1
      features = convert(Array{Float64}, getfeats(model, data))
    else
      features = hcat(features, convert(Array{Float64}, getfeats(model, data)))
    end
  end
  info("Features have been generated!")
  labels = []
  for (x,y) in data
    for i=1:size(y, 2)
      push!(labels, indmax(y[:, i]))
    end
  end
  labels = convert(Array{Float64}, labels)
  if train
    save("$(jldpath)/Features/trnfeats.jld","feats", features)
    save("$(jldpath)/Features/trnlabels.jld","labels", labels)
  else
    save("$(jldpath)/Features/tstfeats.jld","feats", features)
    save("$(jldpath)/Features/tstlabels.jld","labels", labels)
  end
end

function getmeansketch(datatrn)
  count = 0
  meansketch = nothing
  for (x, y) in datatrn
    for i = 1:size(x, 4)
      if meansketch == nothing
        meansketch = x[:, :, :, i]
      else
        meansketch += x[:, :, :, i]
      end
    end
    count += size(x, 4)
  end
  meansketch = meansketch/count
  return meansketch
end

function main(args=ARGS)
  s = ArgParseSettings()
  s.description="SketchANet.jl: Sketch-a-Net that Beats Humans based on https://arxiv.org/abs/1501.07873. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--trained"; action=:store_true; help="check if model is trained")
    ("--readydata"; action=:store_true; help="is data preprocessed and ready")
    ("--readyindx"; action=:store_true; help="check if train and test indices are ready")
    ("--testmode"; action=:store_true; help="true if in test mode")
    ("--pretrained"; action=:store_true; help="true if pretrained model exists")
    ("--featgenmode"; action=:store_true; help="true if in feature generation")
    ("--imsize"; arg_type=Int; default=225; help="Size of input image")
    ("--epochs"; arg_type=Int; default=1; help="Number of epochs for training.")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  dataperclass = 80

  #global batchsize = 50
  #global datapath = "/mnt/kufs/scratch/kkaiyrbekov15/Sketch Network/data"
  lr = 0.001
  global const channels = 6
  if !o[:readyindx]
    info("Indices are not ready")
  end
  if gpu() >= 0
    atype = KnetArray{Float32}
  else
    error("SketchANet only works on GPUs")
  end


  if o[:testmode]
    info("In test mode.")
    if isfile("$(jldpath)/meansketch$(o[:imsize]).jld")
      meansketch = load("$(jldpath)/meansketch$(o[:imsize]).jld")["meansketch"]
      println("meansketch loaded")
    else
      error("No mean sketch file found")
    end
    datatst = load("$(jldpath)/datatst225.jld")["datatst"]
    testaccuracy = test(datatst, meansketch; scales = [225], atype = atype)
    println("Test accuracy: $(testaccuracy)")
  elseif o[:featgenmode]
    info("In feature generating mode.")
    data = load("$(jldpath)/datatst225.jld")["datatst"]
    savefeats(data; scales = [225], train = false)
    data = load("$(jldpath)/datatrn225.jld")["datatrn"]
    savefeats(data; scales = [225], train = true)
  else
    println("In train mode. Imsize=$(o[:imsize])")
    if o[:pretrained]
      w = load("$(jldpath)/Models/FloatModel/tmp/floatmodel$(o[:imsize]).jld")["model"]
      w = map(a->convert(atype, a), w)
      println("Pretrained model loaded!")
    else
      w = weights(atype=atype)
    end
    prms = map(x->Adam(), w)
    global const cmap = initchannelmap()

    #println("Type of o[:imsize] $(typeof(o[:imsize]))")
    if o[:readydata]
      println("Data is ready")
      datatrn = load("$(jldpath)/datatrn$(o[:imsize]).jld")["datatrn"]
    #  datatst = load("$(jldpath)/datatst225.jld")["datatst"]
    else
      println("Data is not ready")
      trnind, tstind = gettraintestindices(dataperclass; readyindx=o[:readyindx])
      datatrn = initdata(trnind; train = true, imsize = o[:imsize])
      datatst = initdata(tstind; train = false, imsize = o[:imsize])
      println("Data minibatched successfully")
    #  save("$(jldpath)/datatrn$(o[:imsize]).jld","datatrn", datatrn)
    #  save("$(jldpath)/datatst$(o[:imsize]).jld","datatst", datatst)
    end

    if isfile("$(jldpath)/meansketch$(o[:imsize]).jld")
      meansketch = load("$(jldpath)/meansketch$(o[:imsize]).jld")["meansketch"]
    else
      meansketch = getmeansketch(datatrn)
      save("$(jldpath)/meansketch$(o[:imsize]).jld","meansketch", meansketch)
    end

    trnlosses = []
    tstlosses = []
    trnerr  = []
    tsterr  = []
    iters = []
    println("Training started, learning rate = $(lr)")
    flush(STDOUT)
    #addprocs(1)
    for i=1:200
      w = trainvanilla(w, prms, datatrn, meansketch; limit = length(datatrn), lr=lr, epochs = 1, atype=atype)
      tstacc, tstloss = 0, 0
      trnacc, trnloss = 0, 0
      if i%25 == 0
        trnacc, trnloss = accuracyvanilla(w, datatrn, meansketch; limit = length(datatrn), atype=atype)
        @printf("epoch: %d trn accuracy-loss: %g - %g tst accuracy-loss: %g - %g\n", i, trnacc, trnloss, tstacc, tstloss)
        flush(STDOUT)
      end
      #tstacc, tstloss = accuracyvanilla(w, datatst; atype=atype)
      if i%100 == 0
        lr = lr/2
      end
    #  append!(trnlosses, trnloss)
  	#	append!(tstlosses, tstloss)
  #		append!(iters, i)
  	#	append!(tsterr, 1-tstacc)
  #		append!(trnerr, 1-trnacc)

      if i%50 == 0
        floatmodel = map(a->convert(Array{Float32}, a), w)
        save("$(jldpath)/tmpSAfloat$(i)model$(o[:imsize]).jld","model", floatmodel)
      end
      #flush(STDOUT)
    end
    #save("$(jldpath)/knetarrmodel$(o[:imsize]).jld","model", w)
    floatmodel = map(a->convert(Array{Float32}, a), w)
    save("$(jldpath)/SAfloatfinalmodel$(o[:imsize]).jld","model", floatmodel)
    save("$(jldpath)/SAknetarrmodel$(o[:imsize]).jld","model", w)
  #  saveplots(iters, trnlosses, tstlosses, trnerr, tsterr, "loss$(o[:imsize])", "err$(o[:imsize])")
  end



end

main()
end
