module SKETCH
#sketch type
type Sketch
  sketch_id
  strokes::Array
  end_indices::Array
end

function addstroke!(sketch::Sketch, stroke::Array)
  #add strokes to sketch and add ending index of the stroke
  sketch.strokes = hcat(sketch.strokes, stroke)
  #if this is first stroke of sketch
  if length(sketch.end_indices) == 0
    sketch.end_indices = hcat(sketch.end_indices, length(stroke))
  else
    prev_end_index = sketch.end_indices[length(sketch.end_indices)]
    sketch.end_indices = hcat(sketch.end_indices, prev_end_index + length(stroke))
  end
  prev_end_index = prev_end_index + length(stroke)
end

function printcontents(sketch::Sketch)
  #Prints contents of current sketch
  start = 1
  strokenum = 1
  println("Contents of sketch with ID = $(sketch.sketch_id) are:")
  #iterate through strokes
  for ending_index in sketch.end_indices
    println("Points of stroke $strokenum")
    #print points of stroke
    for i = start:ending_index
      println("x = $(sketch.strokes[1, i]) y = $(sketch.strokes[2, i])")
    end
    #set new stroke start index
    start = ending_index + 1
    strokenum += 1
  end
end
export Sketch
export printcontents
export addstroke!
end
