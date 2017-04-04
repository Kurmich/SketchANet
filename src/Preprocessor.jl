module Preprocessor
using SKETCH
using PyPlot
using Images
using FileIO
using Colors
using ImageTransformations, CoordinateTransformations
function origin2point!(sketch::Sketch, center)
  sketch.strokes .- center
end

function mirror(sketch::Sketch,  index::Int64, line = 0.0)
  #mirrors x or y points about specific line
  newstrokes = copy(sketch.strokes)
  newid = "$(sketch.sketch_id)mirroredabtx=$line"
  newstrokes[index, :] = 2*line .- newstrokes[index, :]
  return Sketch(newid, newstrokes, sketch.end_indices)
end

function ymirror(sketch::Sketch; line = 0.0)
  return mirror(sketch, 1, line)
end

function xmirror(sketch::Sketch; line = 0.0)
  return mirror(sketch, 2, line)
end

function setrange!(sketch::Sketch, newmax::Number, newmin::Number, index::Number)
  #compute new range
  newrange = newmax - newmin
  #get minimum and maximum of x or y points then compute current range
  min = minimum(sketch.strokes[index, :])
  max = maximum(sketch.strokes[index, :])
  oldrange = max - min
  #change the range
  if oldrange != 0
    sketch.strokes[index, :] = ((sketch.strokes[index, :] .- min)*newrange/oldrange) + newmin
  end
end

function setnewxrange!(sketch::Sketch, newmax::Number, newmin::Number)
  #sets new range for x points of sketch
  setrange!(sketch, newmax, newmin, 1)
end

function setnewyrange!(sketch::Sketch, newmax::Number, newmin::Number)
  #sets new range for y points of sketch
  setrange!(sketch, newmax, newmin, 2)
end

function stretch(sketch::Sketch, stretchfactor::Number, horizontal = true)
  stretchmat = [stretchfactor 0; 0 1]
  if !horizontal
    stretchmat[1, 1] = 1
    stretchmat[2, 2] = stretchfactor
  end
  return applytransformation(sketch, stretchfactor, stretchmat, "stretch")
end

function applytransformation(sketch::Sketch, factor::Number, transmat::Array, name::String)
  #applies specified tranformation to all points if sketch
  newstrokes = transmat*sketch.strokes
  newid = "$(sketch.sketch_id)$(name)by$factor"
  return Sketch(newid, newstrokes, sketch.end_indices)
end

function shear(sketch::Sketch, shearfactor::Number; horizontal = true)
  shearmat = [1 shearfactor; 0 1]
  if !horizontal
    shearmat[1, 2] = 0
    shearmat[2, 1] = shearfactor
  end
  return applytransformation(sketch, shearfactor, shearmat, "shear")
end

function rotate(sketch::Sketch, angle::Number, center=[0 0])
  #convert degrees to radians
  theta = deg2rad(angle)
  #get rotation matrix
  rot = [cos(theta) -sin(theta); sin(theta) cos(theta)]
  #rotate strokes around "center" point
  newstrokes = rot*(sketch.strokes .- (center.')) .+ (center.')
  newid = "$(sketch.sketch_id)rotby$angle"
  #return rotated sketch
  return Sketch(newid, newstrokes, sketch.end_indices)
end

function plotSketch(sketch::Sketch)
  #Prints contents of current sketch
  start = 1
  strokenum = 1
  #iterate through strokes
  for ending_index in sketch.end_indices
    x = sketch.strokes[1, start:ending_index]
    y = sketch.strokes[2, start:ending_index]
    plot(x, y, linewidth = 1)
    #xlim(0, 1000)
    #ylim(0, 200)
    #set new stroke start index
    start = ending_index + 1
    strokenum += 1
  end
end

function saveSketch(sketch::Sketch, filename::String = "test.png"; completness=1, scaled = true)
  #Prints contents of current sketch
  start = 1
  strokenum = 0
  mydpi = 100
  #iterate through strokes
  fig = figure(figsize=(225/mydpi, 225/mydpi), dpi=mydpi, facecolor = "black")
  strokelimit = strokelimit = Integer(ceil(completness*length(sketch.end_indices)))
  for ending_index in sketch.end_indices
    strokenum += 1
    #get points of stroke
    x = []
    y = []
    for i = start:ending_index
      append!(x, sketch.strokes[1, i])
      append!(y, sketch.strokes[2, i])
      #println("x = $(sketch.strokes[1, i]) y = $(sketch.strokes[2, i])")
    end
    if strokenum <= strokelimit
      plot(x, y, linewidth = 1, color = "white")
    else
      plot(x, y, linewidth = 1, color = "black")
    end
    if scaled
      subplots_adjust(bottom=0.,left=0.,right=1.,top=1.)
    end
    axis("off")
    #set new stroke start index
    start = ending_index + 1
  end
  savefig(filename, dpi=mydpi, facecolor= "black")
  close()
end


function bresenham(x1, y1, x2, y2)
	x1 = Integer(round(x1))
	x2 = Integer(round(x2))
	y1 = Integer(round(y1))
	y2 = Integer(round(y2))
	dx = abs(x2-x1)
	dy = abs(y2-y1)
	steep = abs(dy) > abs(dx)
	if steep
		tmp = dx
		dx = dy
		dy = tmp
	end

	if dy == 0
		q = zeros(Int64,dx+1, 1)
	else
		q = [0]
		isOne = diff(mod(collect(floor(dx/2):-dy:-dy*dx+floor(dx/2)), dx)).>=0
		for one in isOne
			if one
				q = vcat(q, 1)
			else
				q = vcat(q, 0)
			end
		end
	end
	#print(q)
	if steep
		if y1 <= y2
			y = collect(y1:y2)
		else
			y = collect(y1:-1:y2)
		end

		if x1 <= x2
			x = x1 .+ cumsum(q)
		else
			x = x1 .- cumsum(q)
		end
	else
		if x1 <= x2
			x = collect(x1:x2)
		else
			x = collect(x1:-1:x2)
		end

		if y1 <= y2
			y = y1 .+ cumsum(q)
		else
			y = y1 .- cumsum(q)
		end

	end

	return x, y

end


function renderSketch!(sketch, image, pixels = nothing)
  #renders stroke to image array
  setnewxrange!(sketch, 350, 50)
  setnewyrange!(sketch, 350, 50)
  if pixels == nothing
    pixels = zeros(length(sketch.strokes))
  end
  start = 1
  for ending_index in sketch.end_indices
    for i = start:(ending_index-1)
      x1 = sketch.strokes[1, i]
      y1 = sketch.strokes[2, i]
      x2 = sketch.strokes[1, i+1]
      y2 = sketch.strokes[2, i+1]
      x, y = bresenham(x1, y1, x2, y2)
      for j = 1:length(x)
        #println( " x = $(x[j]) y = $(y[j])")
        if image[size(image)[1] - y[j], x[j]] > pixels[i]
          image[size(image)[1] - y[j], x[j]] = pixels[i]
        end
      end
    end
    start = ending_index + 1
  end
  #println("sum = """")
  save("newtest.jpg", image)
end

function initvolume!(sketchvol, curbatch, datapath, class, cmap, instancenum, imindex; imsize::Int = 256)
  for c = 1:size(sketchvol, 3)
    path = "$datapath/$(class)/$(cmap[c])/$(instancenum)/$(imindex).png"
    sketchvol[:, :, c, curbatch] = sketchim(path; imsize=imsize)
  end
end

function sketchim(path; imsize::Int = 256, trgsize::Int = 225)
  img0 = load(path)
  if imsize != trgsize
    img = Images.imresize(img0, (imsize,imsize,1))
  else
    img = img0
  end
  #imgSeparated = separate(img)
  imgSeparated = permuteddimsview(channelview(img), (2,3,1))
  imgMatrix = imgSeparated[:, :, 1]
  if size(imgMatrix, 1) != trgsize
    imgMatrix = Images.imresize(imgMatrix, (trgsize, trgsize, 1))
  end
  imgMatrix = convert(Array{Float32}, imgMatrix)
  return imgMatrix
end

function imrotate(img, angle)
  dimnums = length(size(img))
  if dimnums != 3
    throw(ArgumentError("Number of dimensions should be 3. Current dims = $(size(img))"))
  end
  imgnew = zeros(Float32, size(img))
  theta = pi*angle/180 #angle in radians
  tfm = recenter(RotMatrix(theta), center(img[:, :, 1])) #define rotation about center
  for channel=1:size(img, 3)
    imgrot = warp(img[:, :, channel], tfm)
    imgnew[:, :, channel] = imgrot[UnitRange.(indices(img[:, :, channel]))...]
  end
  imgnew[isnan(imgnew)] = 0
  return imgnew
end

export plotSketch
export saveSketch
export translate2origin!
export rotate
export shear
export stretch
export setnewyrange!
export setnewxrange!
export xmirror
export ymirror
export renderSketch!
export imrotate
export sketchim
export initvolume!
end
