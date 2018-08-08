import signal
import datetime
import os
import matplotlib.pyplot as plt
import json

import hail as hl
from hail.plot.slippyplot.zone import Zone, Zones


class TileGenerator(object):

    def __init__(self,
                 mt,
                 dest,
                 empty_tile_path='/Users/maccum/manhattan_data/empty_tile.png',
                 log_path='plot_generation.log',
                 regen=False,
                 x_margin=100000,
                 y_margin=5,
                 bins=256 * 256):
        """
        TileGenerator for tiles (.png images) that compose a plot.

        Format matrix table with hl.experimental.format_manhattan first.

        :param mt: dataset
        :param dest: destination folder
        :param empty_tile_path: path to image of empty tile (whitespace)
        :param log_path: path to logging file
        :param regen: True if tiles with existing files should be regenerated
        :param x_margin: additional space to show on graph outside x range
        :param y_margin: additional space to show on graph outside y range
        """

        self.check_schema(mt)

        self.mt = mt
        self.dest = dest
        if not empty_tile_path:
            self.empty_tile_path = self.dest + "/" + "empty_tile.png"
            self.generate_plot(empty_tile_path, [], [], [], [], [])
        else:
            self.empty_tile_path = empty_tile_path
        self.log_path = log_path
        self.log = None
        self.regen = regen

        # If x and y margins are set to 0, then the graph's boundaries
        # will be exactly equal to the x and y ranges of the plotted data.
        self.x_margin = x_margin
        self.y_margin = y_margin

        # bins for downsample aggregator
        self.bins = bins

        # Empty zones are used to avoid filtering to a region we know is empty.
        # Only useful if multiple zoom levels are generated in succession
        self.empty_zones = Zones()

    # convert global_position to local pixel coordinates
    @staticmethod
    def map_value_onto_range(value, old_range, new_range):
        old_span = old_range[1] - old_range[0]
        new_span = new_range[1] - new_range[0]
        distance_to_value = value - old_range[0]
        percent_span_to_value = distance_to_value / old_span
        distance_to_new_value = percent_span_to_value * new_span
        new_value = new_range[0] + distance_to_new_value
        return new_value

    @staticmethod
    def check_schema(mt):
        # TODO: probably exists a hail util to do this
        row_fields = list(mt.row)
        col_fields = list(mt.col)
        entry_fields = list(mt.entry)
        global_fields = list(mt.globals)

        desired_row = ['global_position', 'color']
        desired_col = ['phenotype', 'max_nlp', 'min_nlp']
        desired_entry = ['neg_log_pval', 'label', 'under_threshold']
        desired_globals = ['gp_range']

        def check_fields(desired, actual, msg):
            for field in desired:
                if field not in actual:
                    raise ValueError(f"For {msg} schema: expected {desired} "
                                     f"but found {actual}")

        (check_fields(desired_row, row_fields, "row")
         and check_fields(desired_col, col_fields, "col")
         and check_fields(desired_entry, entry_fields, "entry")
         and check_fields(desired_globals, global_fields, "global"))

    @staticmethod
    def date():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # run with hail in quiet mode for progress bar to print without re-printing
    # prints progress of tile generation for a layer
    # TODO: probably delete
    @staticmethod
    def progress(i, total, prefix='Zoom Level', suffix='Complete',
                 decimals=1, length=50, fill='█'):
        # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (i / float(total)))
        filled_length = int(length * i // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        if i == total:
            print()

    # filter mt to the desired zone on the graph
    @staticmethod
    def filter_by_coordinates(mt, gp_range, nlp_range, phenotype):
        assert (len(gp_range) == 2 and len(nlp_range) == 2)

        ht = mt.filter_cols(mt.phenotype == phenotype).entries()
        return ht.filter(
            hl.interval(hl.int64(gp_range[0]), hl.int64(gp_range[1]),
                        includes_start=True, includes_end=True).contains(
                ht.global_position) &
            hl.interval(hl.float64(nlp_range[0]), hl.float64(nlp_range[1]),
                        includes_start=True, includes_end=True).contains(
                ht.neg_log_pval)
        )

    # filter table to have one row (variant) per pixel
    # TODO: has been replaced by downsample aggregator; can be deleted
    @staticmethod
    def filter_by_pixel(ht, gp_range, nlp_range):
        assert (len(gp_range) == 2 and len(nlp_range) == 2)
        pixel_coordinates_on_tile = [
            hl.floor(TileGenerator.map_value_onto_range(ht.neg_log_pval,
                                                        nlp_range, [0, 256])),
            hl.floor(TileGenerator.map_value_onto_range(ht.global_position,
                                                        gp_range, [0, 256]))
        ]
        # FIXME : key_by(...).distinct() is slow; use jack's downsampler
        return ht.annotate(
            tile_coordinates=pixel_coordinates_on_tile).key_by(
            'tile_coordinates').distinct()

    # collect global positions, negative log p-values, and colors for matplotlib
    # TODO: not being used, since downsample aggregator returns list, not table
    @staticmethod
    def collect_values(ht):
        global_positions = []
        neg_log_pvals = []
        colors = []
        collected = ht.collect()
        for i in range(0, len(collected)):
            global_positions.append(collected[i].global_position)
            neg_log_pvals.append(collected[i].neg_log_pval)
            colors.append(collected[i].color)
        return global_positions, neg_log_pvals, colors

    # calculate the global position range for a particular tile
    def calculate_gp_range(self, column, num_cols):
        gp_range = self.mt.gp_range.collect()[0]
        x_axis_range = [gp_range.min, gp_range.max]

        x_graph_min = x_axis_range[0] - self.x_margin
        x_graph_max = x_axis_range[1] + self.x_margin
        tile_width = (x_graph_max - x_graph_min) / num_cols
        min_gp = tile_width * column + x_graph_min
        max_gp = min_gp + tile_width
        return [min_gp, max_gp]

    # each phenotype will have its own subdirectory in the self.dest folder
    def directory_name(self, phenotype, zoom):
        return self.dest + "/" + str(phenotype) + "/" + str(zoom)

    # logging file to record as tiles are generated (or skipped if empty)
    def start_log(self, new_log_file, zoom):
        write_method = 'w' if new_log_file else 'a'
        self.log = open(self.log_path, write_method)
        self.log.write("ZOOM "
                       + self.date()
                       + ": generating plots for zoom level : "
                       + str(zoom) + ".\n")

    # flush buffer to the logging file if program is interrupted
    def catch(self, signum, frame):
        if self.log is not None:
            self.log.flush()

    # generate plot and write it to tile_path as a .png
    @staticmethod
    def generate_plot(tile_path, xs, ys, colors, x_range, y_range):
        # figsize=(2.56,2.56) and savefig(dpi=100) => 256 * 256 pixel image
        fig = plt.figure(figsize=(2.56, 2.56))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.scatter(xs, ys, c=colors, s=4)
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)

        plt.savefig(tile_path, dpi=100, transparent=True)
        # plt.show()  # show the plots as they are generated, for debugging
        plt.close()

    # generate a single tile for the given phenotype and ranges of x,y values
    def generate_tile_image(self, zc, x_range, y_range, tile_path, phenotype,
                            json_path, hover_path):
        zone = Zone(x_range, y_range)

        if self.empty_zones.contains(zone):
            # tile will be empty; don't bother filtering table
            self.log.write("EMPTY " + self.date() + ": empty plot <"
                           + tile_path + ">.\n")
            if os.path.exists(tile_path):
                os.unlink(tile_path)
            os.symlink(self.empty_tile_path, tile_path)
            return

        # when filtering by coordinates, should filter slightly outside bounds
        # of image, so that there is a smooth transition between tiles
        # TODO: fine tune how much we need to include outside tile border
        x_span = x_range[1] - x_range[0]
        x_filter_range = [x_range[0] - x_span * .1, x_range[1] + x_span * .1]
        filtered_by_coordinates = self.filter_by_coordinates(self.mt,
                                                             x_filter_range,
                                                             y_range,
                                                             phenotype)

        # this has been replaced by downsample aggregator
        # filtered_by_pixel = self.filter_by_pixel(filtered_by_coordinates,
        #                                         x_range, y_range)

        metadata = (hl.array([filtered_by_coordinates.color,
                              hl.str(filtered_by_coordinates.under_threshold)])
                    .extend(filtered_by_coordinates.label))
        downsampled = filtered_by_coordinates.aggregate(
            hl.agg.downsample(filtered_by_coordinates.global_position,
                              filtered_by_coordinates.neg_log_pval,
                              metadata,
                              n_divisions=self.bins))

        # global positions, negative log p-values, colors, and labels for plot
        gp, nlp, colors = [], [], []
        points = []
        # fixme: slow AF to go through labels one by one after agg.downsample
        for i, elem in enumerate(downsampled):
            gp.append(elem[0])
            nlp.append(elem[1])
            label = elem[2]
            colors.append(label[0])
            if label[1] == "true":
                # p-value is under threshold (nlp over threshold)
                points.append({
                    'gp': elem[0],
                    'nlp': elem[1],
                    'label': label[2:]
                })
        if points:
            # write hover data to json file (note that if threshold was set too wide in format_manhattan(), then this will be extremely slow, and json files will be too big to read
            with open(json_path, 'w') as outfile:
                json.dump(points, outfile)
            with open(hover_path, 'r') as hover_file:
                tiles_with_hover_data = json.load(hover_file)
            with open(hover_path, 'w') as hover_file:
                tiles_with_hover_data.append(zc[1])
                json.dump(tiles_with_hover_data, hover_file)

        if not gp:
            assert not nlp
            # tile is empty; add to empty list
            self.empty_zones.append(zone)
            self.log.write("EMPTY " + self.date() + ": empty plot <"
                           + tile_path + ">.\n")
            if os.path.exists(tile_path):
                os.unlink(tile_path)
            os.symlink(self.empty_tile_path, tile_path)
            return

        self.log.write("GEN " + self.date() + ": generated plot <"
                       + tile_path + ">.\n")

        self.generate_plot(tile_path, gp, nlp, colors, x_range, y_range)

    # generate all tiles for a given phenotype at a particular zoom level
    # if new_log_file=False, log file will be appended to, not overwritten
    def generate_tile_layer(self, phenotype, zoom, new_log_file=False):
        signal.signal(signal.SIGINT, self.catch)

        zoom_directory = self.directory_name(phenotype, zoom)

        if not os.path.exists(zoom_directory):
            os.makedirs(zoom_directory)

        self.start_log(new_log_file, zoom)

        pheno_info = self.mt.filter_cols(
            self.mt.phenotype == phenotype).cols().collect()[0]
        y_axis_range = [pheno_info.min_nlp - self.y_margin,
                        pheno_info.max_nlp + self.y_margin]

        # write x and y axis bounds to log
        gp_range = self.mt.gp_range.collect()[0]
        x_graph_min = gp_range.min - self.x_margin
        x_graph_max = gp_range.max + self.x_margin
        self.log.write(
            "AXIS BOUNDS: x[{},{}] y[{},{}]\n".format(x_graph_min, x_graph_max,
                                                      y_axis_range[0],
                                                      y_axis_range[1]))

        metadata_file = open(
            self.dest + "/" + phenotype + "/" + "metadata.json", 'w')
        json.dump({'phenotype': phenotype,
                   'x_axis_range': [x_graph_min, x_graph_max],
                   'y_axis_range': y_axis_range}, metadata_file)
        metadata_file.close()

        hover_path = zoom_directory + "/" + "hover.json"
        with open(hover_path, 'w') as hover_file:
            json.dump([], hover_file)

        num_cols = 2 ** zoom
        iteration = 1
        for c in range(0, num_cols):
            x_range = self.calculate_gp_range(c, num_cols)

            json_path = zoom_directory + "/" + str(c) + ".json"
            tile_path = zoom_directory + "/" + str(c) + ".png"
            if (not os.path.isfile(tile_path)) or self.regen:
                zc = [zoom, c]
                self.generate_tile_image(zc, x_range, y_axis_range,
                                         tile_path, phenotype, json_path,
                                         hover_path)
                self.progress(iteration, num_cols,
                              prefix='Zoom level: ' + str(zoom))

            iteration = iteration + 1
        hover_file.close()
        self.log.close()
