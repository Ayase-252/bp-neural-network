function coordiates = coord_generator(x, y)
    [x_1, y_1] = meshgrid(x, y);
    coordiates = zeros(2, numel(x_1));
    for i = 1 : numel(x_1)
        coordiates(:, i) = [x_1(i); y_1(i)];
    end
end

