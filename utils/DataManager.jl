include("Sketch.jl")
include("Preprocessor.jl")
include("SketchGetter.jl")

module DataManager
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
  dataind -> array of all file indices; range = [1,20000]
  imgindx -> index of an augmented image; range = [1, # of tranformations]
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
  #define data and labels for a batch
  data = Array{Float32}(trgsize, trgsize, channels, len)
  labels = zeros(Float32, classnum, len)
  #initialize each instance and label of a batch
  fillbatch!(data, labels, imindex, imsize, batchind, len)
  return (data, labels)
end

function fillbatch!(data, labels, imindex, imsize::Int, batchind, len)
  for i=1:len
    filenum = batchind[i]
    class, instancenum = filenum2ind(filenum)
    labels[class, i] = 1
    initvolume!(data, i, datapath, class, cmap, instancenum, imindex; imsize = imsize)
  end
end

function split(dataperclass; trainpercent::Float64 = 0.67, classnum::Int = 250, N::Int = 80, readyindx::Bool = false)
  #number of training samples per classs
  traincount = Int(round(trainpercent*dataperclass))
  first  = Int(round(traincount/2))
  #random permutation of indexes in single class
  perm = randperm(dataperclass)
  #file indices for splits 1,2,3
  part1 = perm[1:first]
  part2 = perm[first+1:traincount]
  part3 = perm[traincount+1:end]
  part1cpy = copy(part1)
  part2cpy = copy(part2)
  part3cpy = copy(part3)
  #add corresponding entries of each class
  for c=2:classnum
    part1 = vcat(part1, (c-1)*N .+ part1cpy)
    part2 = vcat(part2, (c-1)*N .+ part2cpy)
    part3 = vcat(part3, (c-1)*N .+ part3cpy)
  end
  part1 = shuffle(part1)
  part2 = shuffle(part2)
  part3 = shuffle(part3)
  return part1, part2, part3
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

function savedata(dataperclass, partition; imsize::Int = 225, readyindx::Bool = false)
  println("Data is not ready")
  if !readyindx
    println("Indices are not ready")
    part1, part2, part3 = split(dataperclass; readyindx=o[:readyindx])

  end

  trnind_1, tstind_1 = vcat(part1, part2), part3
  trnind_2, tstind_2 = vcat(part1, part3), part2
  trnind_3, tstind_3 = vcat(part2, part3), part1
  datatrn = initdata(trnind; train = true, imsize = o[:imsize])
  datatst = initdata(tstind; train = false, imsize = o[:imsize])
  println("Data minibatched successfully")
end


function main(args=ARGS)
  s = ArgParseSettings()
  s.description="My Model. (c) Kurmanbek Kaiyrbekov 2017."
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin
    ("--readydata"; action=:store_true; help="Is the data preprocessed and ready?")
    ("--readyindx"; action=:store_true; help="Check if train and test indices are ready")
    ("--imsize"; arg_type=Int; default=225; help="Size of input image")
    ("--partition"; arg_type=Int; default=1; help="Number of parts")
  end
  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  dataperclass = 80
  lr = 0.01
  global const channels = 6
  global const cmap = initchannelmap()
  if !o[:readyindx]
    info("Indices are not ready")
  end
end

end
