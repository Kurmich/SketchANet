module SketchGetter
using SKETCH
using Preprocessor
#import Preprocessor.translate2origin!
function geteitzsketches(class; completness=1)
  foldername = "/home/kurmanbek/Desktop/EitzSketches/"
  from = (class-1) * 80 + 1
  sketches = Any[]
  extension = ".txt"
  for foldernum = class:class
    to = from + 79
    for filenum = from:to
      filename =  "$foldername/$foldernum/$filenum$extension"
      fid = open(filename)
      sketch = txt2sketch(fid, filenum; completness=completness)
      push!(sketches, sketch)
      #push!(sketches, sketch)
      #translate2origin!(sketch)
      #printcontents(sketch)
      #close(fid)
    end
    from = to + 1
  end
  return sketches
end

function isnumeric{T<:String}(s::T)
    isa(parse(s), Number)
end

function txt2sketch(fid,  sketchid; completness=1)
  strokes = Float64[]
  indices = Int[]
  #Read all lines
  lines  = readlines(fid)
  prev_end_index = 0
  strokecount = 0
  strokelimit = 1
  #traverse lines one by one
  for ln in lines
    splt = split(ln)
    if splt[1] == "Numstrokes"
      strokelimit = parse(Int, splt[2])
      strokelimit = Integer(ceil(completness*strokelimit))
    end
    if splt[1] == "Stroke"
      strokecount += 1
      if strokecount > strokelimit
        break
      end
      numpoints = parse(Int, splt[4])
      cur_end_index = prev_end_index + numpoints
      append!(indices, cur_end_index)
      prev_end_index = cur_end_index
    end
    if isa(parse(splt[1]), Number)
      append!(strokes, parse(Float64, splt[1]))
      append!(strokes, parse(Float64, splt[2]))
    end
  end

  strokes = reshape(strokes, (2, convert(Int64, length(strokes)/2)))
  #println(indices)
  return Sketch("$sketchid", strokes, indices)
end
export geteitzsketches
end
