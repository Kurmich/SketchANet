function bresenham(x1, y1, x2, y2)
	x1 = round(x1)
	x2 = round(x2)
	y1 = round(y1)
	y2 = round(y2)
	dx = abs(x2-x1)
	dy = abs(y2-y1)
	steep = abs(dy) > abs(dx)
	if steep
		tmp = dx
		dx = dy
		dy = tmp
	end

	if dy == 0
		q = zeros(dx+1, 1)
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
	print(q)
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

x, y = bresenham(1, 2, -1, -8)
