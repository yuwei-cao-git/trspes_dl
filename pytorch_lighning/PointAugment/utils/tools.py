import glob
import os
from datetime import datetime
from pathlib import Path
from itertools import cycle, islice

import laspy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import torch
#from plyer import notification
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)

    # Get coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        # for las_field in las_fields[3:]:  # skip the X,Y,Z fields
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)


def farthest_point_sampling(coords, k):
    # Adapted from https://minibatchai.com/sampling/2021/08/07/FPS.html

    # Get points into numpy array
    points = np.array(coords)

    # Get points index values
    idx = np.arange(len(coords))

    # Initialize use_idx
    use_idx = np.zeros(k, dtype="int")

    # Initialize dists
    dists = np.ones_like(idx) * float("inf")

    # Select a point from its index
    selected = 0
    use_idx[0] = idx[selected]

    # Delete Selected
    idx = np.delete(idx, selected)

    # Iteratively select points for a maximum of k samples
    for i in range(1, k):
        # Find distance to last added point and all others
        last_added = use_idx[i - 1]  # get last added point
        dist_to_last_added_point = ((points[last_added] - points[idx]) ** 2).sum(-1)

        # Update dists
        dists[idx] = np.minimum(dist_to_last_added_point, dists[idx])

        # Select point with largest distance
        selected = np.argmax(dists[idx])
        use_idx[i] = idx[selected]

        # Update idx
        idx = np.delete(idx, selected)
    return use_idx



class PointCloudsInPickle(Dataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(
        self,
        filepath,
        pickle,
        # column_name="",
        # max_points=200_000,
        # samp_meth=None,  # one of None, "fps", or "random"
        # use_columns=None,
    ):
        """
        Args:
            pickle (string): Path to pickle dataframe
            column_name (string): Column name to use as target variable (e.g. "Classification")
            use_columns (list[string]): Column names to add as additional input
        """
        self.filepath = filepath
        self.pickle = pd.read_pickle(pickle)
        # self.column_name = column_name
        # self.max_points = max_points
        # if use_columns is None:
        #     use_columns = []
        # self.use_columns = use_columns
        # self.samp_meth = samp_meth
        super().__init__()

    def __len__(self):
        return len(self.pickle)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file name
        pickle_idx = self.pickle.iloc[idx : idx + 1]
        filename = pickle_idx["FileName"].item()

        # Get file path
        file = os.path.join(self.filepath, filename)

        # Read las/laz file
        coords = read_las(file, get_attributes=False)

        # # Resample number of points to max_points
        # if self.samp_meth is None:
        #     use_idx = np.arange(len(coords))
        # else:
        #     if coords.shape[0] >= self.max_points:
        #         if self.samp_meth == "random":
        #             use_idx = np.random.choice(
        #                 coords.shape[0], self.max_points, replace=False
        #             )
        #         if self.samp_meth == "fps":
        #             use_idx = farthest_point_sampling(coords, self.max_points)
        #     else:
        #         use_idx = np.random.choice(
        #             coords.shape[0], self.max_points, replace=True
        #         )

        # # Get x values
        # if len(self.use_columns) > 0:
        #     x = np.empty((self.max_points, len(self.use_columns)), np.float32)
        #     for eix, entry in enumerate(self.use_columns):
        #         x[:, eix] = attrs[entry][use_idx]
        # else:
        #     # x = coords[use_idx, :]
        #     x = None
        coords = coords - np.mean(coords, axis=0)  # centralize coordinates

        # impute target
        target = pickle_idx["perc_specs"].item()
        target = target.replace("[", "")
        target = target.replace("]", "")
        target = target.split(",")
        target = [float(i) for i in target]  # convert items in target to float

        # if x is None:
        #     sample = Data(
        #         x=None,
        #         y=torch.from_numpy(np.array(target)).type(torch.FloatTensor),
        #         pos=torch.from_numpy(coords[use_idx, :]).float(),
        #     )
        # else:
        #     sample = Data(
        #         x=torch.from_numpy(x).float(),
        #         y=torch.from_numpy(np.array(target)).type(torch.FloatTensor),
        #         pos=torch.from_numpy(coords[use_idx, :]).float(),
        #     )
        coords = torch.from_numpy(coords).float()
        target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
        # target = torch.from_numpy(np.array(target)).half()
        if coords.shape[0] < 100:
            return None
        return coords, target


def write_las(outpoints, outfilepath, attribute_dict={}):
    """
    :param outpoints: 3D array of points to be written to output file
    :param outfilepath: specification of output file (format: las or laz)
    :param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    :return: None
    """
    import laspy

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.x_scale = 0.00025
    hdr.y_scale = 0.00025
    hdr.z_scale = 0.00025
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    las = laspy.LasData(hdr)

    las.x = outpoints[0]
    las.y = outpoints[1]
    las.z = outpoints[2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(vals[0])))
            las[key] = vals

    las.write(outfilepath)
    
class IOStream:
    # Adapted from https://github.com/vinits5/learning3d/blob/master/examples/train_pointnet.py
    def __init__(self, path):
        # Open file in append
        self.f = open(path, "a")

    def cprint(self, text):
        # Print and write text to file
        print(text)  # print text
        self.f.write(text + "\n")  # write text and new line
        self.f.flush  # flush file

    def close(self):
        self.f.close()  # close file


def _init_(model_name):
    # Create folder structure
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + model_name):
        os.makedirs("checkpoints/" + model_name)
    if not os.path.exists("checkpoints/" + model_name + "/models"):
        os.makedirs("checkpoints/" + model_name + "/models")
    if not os.path.exists("checkpoints/" + model_name + "/output"):
        os.makedirs("checkpoints/" + model_name + "/output")
    if not os.path.exists("checkpoints/" + model_name + "/output/laz"):
        os.makedirs("checkpoints/" + model_name + "/output/laz")


def make_confusion_matrix(
    # Adapted from https://github.com/DTrimarchi10/confusion_matrix
    
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    percent:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def delete_files(root_dir, glob="*"):
    # List files in root_dir with glob
    files = list(Path(root_dir).glob(glob))

    # Delete files
    for f in files:
        os.remove(f)


def plot_3d(coords, save_plot=False, fig_path=None, dpi=300):
    # Plot parameters
    plt.rcParams["figure.figsize"] = [7.00, 7.00]  # figure size
    plt.rcParams["figure.autolayout"] = True  # auto layout

    # Create figure
    fig = plt.figure()  # initialize figure
    ax = fig.add_subplot(111, projection="3d")  # 3d projection
    x = coords[:, 0]  # x coordinates
    y = coords[:, 1]  # y coordinates
    z = coords[:, 2]  # z coordinates
    ax.scatter(x, y, z, c=z, alpha=1)  # create a scatter plot
    # plt.show()  # show plot

    if save_plot is True:
        plt.savefig(fig_path, bbox_inches="tight", dpi=dpi)
        

def plot_2d(coords, out_name=None, save_fig=True):
    # Plot parameters
    plt.rcParams["figure.figsize"] = [7.00, 7.00]  # figure size
    plt.rcParams["figure.autolayout"] = True  # auto layout

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Remove axes
    ax.set_axis_off()

    # Create CMAP
    a = 0.75
    my_cmap = plt.cm.rainbow(np.arange(plt.cm.rainbow.N))
    my_cmap[:, 0:3] *= a
    my_cmap = ListedColormap(my_cmap)

    # Get coordinates
    x = coords[0]  # x coordinates
    y = coords[1]  # y coordinates
    z = coords[2]  # z coordinates
    ax.scatter(x, z, s=3, c=z, cmap=my_cmap, alpha=1)  # create a scatter plot

    if save_fig is True and out_name is not None:
        plt.savefig(out_name, bbox_inches="tight", dpi=600)
    plt.close()
    
    
def check_multi_gpu(n_gpus):
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1 and n_gpus > 1:
        multi_gpu = True
        print("Using Multiple GPUs")
    else:
        multi_gpu = False
        
    return multi_gpu


def create_empty_df():
    # Create an empty dataframe with specific dtype
    df = pd.DataFrame(
        {
            "Model": pd.Series(dtype="str"),
            "Point Density": pd.Series(dtype="str"),
            "Overall Accuracy": pd.Series(dtype="float"),
            "F1": pd.Series(dtype="float"),
            "Augmentation": pd.Series(dtype="str"),
            "Sampling Method": pd.Series(dtype="str"),
        }
    )

    return df


def variable_df(variables, col_names):
    # Create a dataframe from list of variables
    df = pd.DataFrame([variables], columns=col_names)
    
    return df


def concat_df(df_list):
    # Concate multiple dataframes in list
    df = pd.concat(df_list, ignore_index=True)
    return df

'''
def notifi(title, message):
    # Creates a pop-up notification
    notification.notify(title=title, message=message, timeout=10)
'''
    
    
def create_comp_csv(y_true, y_pred, classes, filepath):
    # Create a CSV of the true and predicted species proportions
    classes = cycle(classes) # cycle classes
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}) # create dataframe
    df["SpeciesName"] = list(islice(classes, len(df))) # repeat list of classes
    species = df.pop("SpeciesName") # remove species name
    df.insert(0, "SpeciesName", species) # add species name
    df.to_csv(filepath, index=False) # save to csv
    
    
def get_stats(df):
    # Get stats
    r2 = r2_score(df["y_true"], df["y_pred"])  # r2
    rmse = np.sqrt(mean_squared_error(df["y_true"], df["y_pred"]))  # rmse
    df_max = np.max(df["Difference"])  # max difference
    df_min = np.min(df["Difference"])  # min difference
    df_std = np.std(df["Difference"])  # std difference
    df_var = np.var(df["Difference"])  # var difference
    df_count = np.count_nonzero(df["y_true"])  # count of none 0 y_true

    return pd.Series(
        dict(
            r2=r2,
            rmse=rmse,
            maximum=df_max,
            minimum=df_min,
            std=df_std,
            var=df_var,
            count=df_count,
        )
    )


def get_df_stats(csv, min_dif, max_dif):
    # Create dataframe
    df = pd.read_csv(csv)  # read in csv
    df["Difference"] = df["y_pred"] - df["y_true"]  # difference of y_pred and y_true
    df["Sum"] = df["y_pred"] + df["y_true"]  # sum of y_pred and y_true
    df["Between"] = df["Difference"].between(min_dif, max_dif)  # boolean of Difference

    # Print number of True and False values of min and max difference values
    print("Count")
    print(df.groupby("Between")["Difference"].count())
    print()

    # Calculate and print get_stats fpr df
    print("Stats")
    df_stats = df.groupby("SpeciesName").apply(get_stats)
    print(df_stats)
    print("#################################################")
    print()

    return df_stats


def scatter_plot(csv, point_density, root_dir):
    df = pd.read_csv(csv)
    species = df.SpeciesName.unique()
    for i in species:
        species_csv = df[df.SpeciesName == i]
        species_csv.plot.scatter(x="y_pred", y="y_true")
        plt.title(f"{i}: {point_density}")
        plt.savefig(os.path.join(root_dir, f"{i}_{point_density}.png"))
        plt.close()


def plot_stats(
    root_dir,
    point_densities,
    model,
    stats=["r2", "rmse"],
    save_csv=False,
    csv_name=None,
    save_fig=False,
):
    # Set Plot Parameters
    plt.rcParams["figure.figsize"] = [15.00, 7.00]  # figure size
    plt.rcParams["figure.autolayout"] = True  # auto layout
    plt.rcParams["figure.facecolor"] = "white"  # facecolor

    dfs_r2 = []
    dfs_rmse = []
    for x in point_densities:
        # Print point density
        print(f"Point Density: {str(x)}")

        # Get root directory
        model_output = os.path.join(root_dir, f"{model}_{x}\output")

        # Get CSVs
        csv = list(Path(model_output).glob("outputs*.csv"))

        # Iterate through CSVs
        for y in csv:
            # Create scatter plots
            scatter_plot(y, x, model_output)

            # Calculate stats
            csv_stats = get_df_stats(y, -0.05, 0.05)

            # Save csv
            if save_csv is True:
                csv_stats.to_csv(
                    os.path.join(model_output, f"{model}_{x}_{csv_name}"), index=False
                )

        for stat in stats:
            # Convert to dataframe
            csv_item = csv_stats[stat].to_frame()

            # Rename column to point denisty
            csv_item.rename({stat: x}, axis=1, inplace=True)

            # Append dfs list
            if stat == "r2":
                dfs_r2.append(csv_item)
            if stat == "rmse":
                dfs_rmse.append(csv_item)

    # Concatenate dataframes
    df_r2 = pd.concat(dfs_r2, axis=1)
    df_rmse = pd.concat(dfs_rmse, axis=1)

    # Create Bar Chart for r2
    df_r2.plot.bar(width=0.9, edgecolor="black")
    plt.ylabel("r2")
    plt.grid(color="grey", linestyle="--", linewidth=0.1)
    plt.legend(title="Point Density")
    plt.tight_layout()

    # Save Figure
    if save_fig is True:
        plt.savefig(os.path.join(root_dir, f"{model}_r2.png"))
    plt.close()

    # Create Bar Chart for rmse
    df_rmse.plot.bar(width=0.9, edgecolor="black")
    plt.ylabel("rmse")
    plt.grid(color="grey", linestyle="--", linewidth=0.1)
    plt.legend(title="Point Density")
    plt.tight_layout()

    # Save Figure
    if save_fig is True:
        plt.savefig(os.path.join(root_dir, f"{model}_rmse.png"))
    plt.close()