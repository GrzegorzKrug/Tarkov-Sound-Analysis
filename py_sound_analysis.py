import pandas as pd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import glob
import os

from scipy.io import wavfile
from matplotlib.style import use
from scipy.interpolate import interp1d
from scipy.signal import spectrogram

from matplotlib.style import use
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
# from matplotlib.lines import MarkerStyle
from matplotlib.collections import PathCollection


headsets = ['M32', 'Gssh', 'CT2', 'CT4', 'Razor', 'Excel', 'Sports', 'Sordin', 'RAC']
postfix = ['-' + name for name in headsets]
postfix.insert(0, '')

COLORS = [
        (0, 0.9, 0),  # green
        (0, 0.6, 0.9),  # blue
        (0, 0.1, 0.4),  # dark blue
        (0.4, 0.2, 0),  # brown
        (0.5, 0.5, 0.5),  # gray
        (0.6, 0.1, 0.7),  # purple
        (0.7, 0.4, 0.2),  # orange
        (0.9, 0.8, 0),  # yellow
        (0.3, 0.7, 0.1),  # GrayGreen
        (1, 0.2, 0.3),  # strawbery
]

TITLE = {
        "Labs": "Labs ambient sound",
        "LabsWalk": "Walking on Labs floor near office",
        "LabsRun": "Running on Labs floor near office",
        "Ambient": "Customs ambient sound",
        "Rain": "Rain sound",
        "HeavyRain": "Heavy rain sound",
        "SV": "SV98 shots on labs",
        "Grass": "Grass walking on customs",
        "Bush": "Walking through trees on customs",
        "Wood": "Walking on wooden floor in shack on 'woods'",
}

tests = [
        'Ambient', 'Labs', 'Rain', 'HeavyRain',
        'Grass', 'Wood', 'LabsWalk', 'LabsRun',
        'Bush', 'SV'
]

FOLDER = "Samples" + os.path.sep
FIG_SIZE = 19, 15
LB_SIZE = 9

use('ggplot')


def calc_average_amp(signal):
    """Average amplitude"""
    mean = np.mean(np.abs(signal))
    return mean


def smooth_conv(mono, N=256, mode='valid'):
    smooth_mean = np.convolve(mono, 1 / N * np.ones(N), mode=mode)
    return smooth_mean


import warnings


warnings.filterwarnings("ignore")


def read_mono(path):
    fz, true_signal = wavfile.read(path)

    mono = true_signal.mean(axis=1)
    return fz, mono


def calc_int_to_dB(int, bits=16):
    full = 2 ** bits
    amp = int / full
    dbval = 10 * np.log10(amp)
    return dbval


def plot_smooth_graph(intervals):
    os.makedirs("plots", exist_ok=True)
    os.makedirs("smooth_wav", exist_ok=True)
    for ti, test in enumerate(tests):
        fig_smooth = plt.figure(figsize=(12, 9), dpi=250)
        plt.title(test)
        for hi, head in enumerate(postfix):
            path = FOLDER + test + head + ".wav"
            plt.subplot(4, 3, hi + 1)
            if not os.path.isfile(path):
                print("notfound:", path)
                continue
            fz, mono = read_mono(path)
            mono = np.abs(mono)
            # amp = calc_int_to_dB(mono)
            head = head[1:] if head else "None"

            for i, inter in enumerate(intervals):
                N = np.round(fz * inter / 1000).astype(int)
                smth = smooth_conv(mono, N)
                plt.plot(smth, label=f"{inter}ms", color=COLORS[i])

            plt.title(head)
            plt.legend()
            plt.ylabel("volume")
            plt.xlabel("time")
            ax = plt.gca()
            # ax.set_xticklabels()
            ax.set_xticks([])

        plt.suptitle(test, fontsize=20)
        plt.tight_layout()
        plt.savefig(f"plots{os.path.sep}{test}.png")
        print(f"Test {test} completed.")


import matplotlib as mpl


class LogarithmCmap(mpl.colors.Colormap):
    """ Colormap adaptor that uses another Normalize instance
    for the colormap than applied to the mappable. """

    def __init__(self, base, cmap_norm, orig_norm=None):
        if orig_norm is None:
            if isinstance(base, mpl.cm.ScalarMappable):
                orig_norm = base.norm
                base = base.cmap
            else:
                orig_norm = mpl.colors.Normalize(0, 1)
        self._base = base
        if (
                isinstance(cmap_norm, type(mpl.colors.Normalize))
                and issubclass(cmap_norm, mpl.colors.Normalize)
        ):
            # a class was provided instead of an instance. create an instance
            # with the same limits.
            cmap_norm = cmap_norm(orig_norm.vmin, orig_norm.vmax)
        self._cmap_norm = cmap_norm
        self._orig_norm = orig_norm

    def __call__(self, X, **kwargs):
        """ Re-normalise the values before applying the colormap. """
        return self._base(self._cmap_norm(self._orig_norm.inverse(X)), **kwargs)

    def __getattr__(self, attr):
        """ Any other attribute, we simply dispatch to the underlying cmap. """
        return getattr(self._base, attr)


def plot_spectograms(intervals):
    os.makedirs("spectograms", exist_ok=True)

    cmap = LogarithmCmap(mpl.cm.coolwarm, mpl.colors.LogNorm(1e-5, 2e-2))
    cmap = LogarithmCmap(mpl.cm.coolwarm, mpl.colors.LogNorm(1e-6, 1e-2))
    # cmap = LogarithmCmap(mpl.cm.coolwarm, mpl.colors.Normalize(1e-4, 1e-2))

    for ti, test in enumerate(tests):
        fig_smooth = plt.figure(figsize=(16, 8), dpi=300)
        # if "labs" not in test.lower():
        #     continue
        for hi, head in enumerate(postfix):
            "Prepare subplot"
            plt.subplot(2, 5, hi + 1)

            "Read file"
            path = FOLDER + test + head + ".wav"
            if not os.path.isfile(path):
                print("\t Not found:", path)
                continue
            # fz, mono = read_mono(path)
            fz, true_signal = wavfile.read(path)
            mono, _ = true_signal.T
            mono = mono[44000:(7 * 44000)]
            mono = mono / mono.max()
            # mono = mono - mono.min()
            # mono = mono * (2 ** 16 / mono.max())
            # mono = np.round(mono).astype(int)
            # plt.figure()
            # plt.plot(mono)
            # plt.show()


            # mono = np.abs(mono)
            head = head[1:] if head else "None"

            # for i, inter in enumerate(intervals):
            #     N = np.round(fz * inter / 1000).astype(int)
            #     smth = smooth_conv(mono, N)
            #     plt.plot(smth, label=f"{inter}ms", color=COLORS[i])

            freq, times, Sxx = spectrogram(
                    mono, window=('tukey', 0.001), fs=fz,
                    nperseg=44 * 20, nfft=2000,
                    # scaling='spectrum',
                    # noverlap=True,
            )
            # print(Sxx.min(), np.median(Sxx), Sxx.max())
            # min_, med, max_ = Sxx.min(), np.median(Sxx), Sxx.max()

            # qlow, qhigh = np.quantile(Sxx.ravel(), [0.5, 1])
            # print()
            # print(med)
            # print(q1, q3)
            # space = np.geomspace(1e-12, 1e-3, 100)
            # mask = Sxx < 0.0001
            # Sxx[mask] = 0.0001


            # plt.hist(Sxx.ravel(), bins=space)
            # plt.ylim([-5, 10000])
            # plt.xticks(rotation=30)

            plt.pcolormesh(times, freq, Sxx, cmap=cmap)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            # plt.ylim(0, 5000)

            plt.title(f"{head}")

        plt.suptitle(f"Spectrograms: {TITLE[test]}", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"spectograms{os.path.sep}{test}.png")
        print(f"Test {test} completed.")


FZ = 44100
MARKER_SIZE = 30


def three_one_plot():
    """
    Combined plot.
    Average
    Peak
    Peak smooth

    """
    # smooth_intervals = np.array([0.001, 0.005, 0.025, 0.1, 0.2])
    smooth_intervals = np.array([1, 5, 15, 25, 50]) / 1000

    ev = np.logspace(0.001, 0.1, 5)
    # smooth_intervals = np.log10(ev).round(3)
    # smooth_intervals = np.linspace(0.001, 0.1, 5).round(3)

    print(smooth_intervals * 1000)
    MARKERS = ['P', '*', 's', 'o', 7]
    MARKERS = [MARKERS[i] if (i + 1) < len(MARKERS) else "." for i in range(len(smooth_intervals))]
    MARKERS[-1] = 7

    mli = pd.MultiIndex(
            levels=[[], []],
            codes=[[], []],
            names=['headset', 'measure'],
    )
    if os.path.isfile("combined.csv"):
        df = pd.read_csv("combined.csv", index_col=[0, 1])
    else:
        df = pd.DataFrame(columns=[t for t in tests], index=mli)

    fig = plt.figure(figsize=(16, 10), dpi=200)
    for ti, test in enumerate(tests):
        print(test)
        plt.subplot(3, 4, ti + 1)
        title = TITLE[test]
        plt.title(title)
        for hi, head in enumerate(postfix):
            path = FOLDER + test + head + ".wav"
            if not os.path.isfile(path):
                print("notfound:", path)
                continue

            if head == "":
                head = "None"
            else:
                head = head[1:]

            fz, mono = read_mono(path)

            avg = calc_average_amp(mono)
            peak = np.max(np.abs(mono))
            avg_db = calc_int_to_dB(avg)
            peak_db = calc_int_to_dB(peak)

            df.loc[(head, "average"), test] = avg_db
            df.loc[(head, "peak"), test] = peak_db

            x1 = hi / 10
            x2 = x1 + 1 / 11
            plt.plot([x1, x2], [avg_db, avg_db], c=COLORS[hi], label=head, linewidth=3)

            plt.scatter(
                    np.mean([x1, x2]), peak_db, s=MARKER_SIZE, marker=6,
                    color=COLORS[hi], edgecolors='black', linewidth=1,
            )

            for i, interval in enumerate(smooth_intervals):
                "Calc new convolution"

                if df.index.isin([(head, str(interval))]).any():
                    val = df.loc[(head, str(interval)), test]
                    if val is np.nan:
                        read_val = np.nan
                    else:
                        read_val = df.loc[(head, str(interval)), test]
                else:
                    read_val = np.nan

                if np.isnan(read_val):
                    N = np.round(interval * FZ).astype(int)
                    smooth_mono = smooth_conv(np.abs(mono), N)
                    smooth_peak = np.max(smooth_mono)
                    peak_smooth_db = calc_int_to_dB(smooth_peak)
                    df.loc[(head, str(interval)), test] = peak_smooth_db
                else:
                    peak_smooth_db = read_val

                plt.scatter(
                        np.mean([x1, x2]), peak_smooth_db, s=MARKER_SIZE, marker=MARKERS[i],
                        color=COLORS[hi],
                        edgecolors='black',
                        # markerfacecolor='red',
                        linewidth=0.5,
                )

        plt.ylabel("dB")
        plt.tight_layout()

    # ax = plt.gca()
    # handles, labels = ax.get_legend_handles_labels()
    # handles, labels = fig.get_legend_handles_labels()
    plt.figure()

    handles = [
            Line2D([0, 0], [0, 0], color=(0, 0, 0)),
            plt.scatter(0, 0, c=(0, 0, 0), marker=6)
    ]
    labels = [
            "Average",
            "Peak"
    ]

    for i, interval in enumerate(smooth_intervals):
        sym = plt.scatter(0, 0, c=(0, 0, 0), marker=MARKERS[i])
        handles.append(sym)
        labels.append(f"Peak (smoothing {int(interval * 1000)}ms)")

    for i, head in enumerate(postfix):
        if head == "":
            head = "None"
        else:
            head = head[1:]

        handles.append(Line2D([0, 0], [0, 0], color=COLORS[i], linewidth=5))
        labels.append(head)

    fig.legend(handles, labels, loc='lower right')
    fig.suptitle(
            f"Hearing own noises. Higher = Louder",
            fontsize=20)
    fig.tight_layout()
    fig.savefig("combined.png")

    plt.figure(fig)
    for ti, t in enumerate(tests):
        plt.subplot(3, 4, ti + 1)
        plt.ylim([-34, -3])
        # plt.ylim([-35, -15])

    fig.savefig("combined_same_scale.png")
    df = df.round(4)
    df.to_csv("combined.csv")


def headset_iterator():
    for ti, test in enumerate(tests):
        for hi, head in enumerate(postfix):
            yield (ti, test), (hi, head)


def pmc_action_iterator():
    white = ["Grass", 'Bush', 'Wood', "LabsWalk", "LabsRun", "SV"]
    white_test = [t for t in tests if t in white]
    for ti, test in enumerate(white_test):
        for hi, head in enumerate(postfix):
            # if head == "":
            #     continue
            yield (ti, test), (hi, head)


AMBIENT_REFERENCE = {
        "Bush": "Ambient",
        "LabsWalk": "Labs",
        "LabsRun": "Labs",
        "Wood": "Ambient",
        "SV": "Labs",
        "Grass": "Ambient",

}


def compare_spreads(long='average', group_headphones=False):
    df = pd.read_csv("combined.csv", index_col=[0, 1])

    stack_intervals = ['peak', 1, 5, 15, 25, 50]
    if long == "average":
        long_key = long
    else:
        long_key = str(long / 1000)

    fig = plt.figure(figsize=(16, 10), dpi=250)
    for i, inter in enumerate(stack_intervals):
        if isinstance(inter, str):
            short_key = inter
        else:
            short_key = str(inter / 1000)

        for (ti, test), (hi, head) in pmc_action_iterator():
            if head == "":
                head = "None"
            else:
                head = head[1:]

            db1 = df.loc[(head, short_key), test]
            db2 = df.loc[(head, long_key), AMBIENT_REFERENCE[test]]

            val1 = (10 ** (db1 / 10))
            val2 = (10 ** (db2 / 10))

            diff = val1 - val2
            # print(diff)

            db_diff = np.abs(10 * np.log10(diff / val2))

            "Plotting bars"
            plt.subplot(2, 3, ti + 1)
            title = TITLE[test]
            plt.title(title)
            if group_headphones:
                width = 1 / (len(stack_intervals) + 1)
                x1 = hi + i * width
            else:
                x1 = i + hi / 11
                width = 1 / 11

            plt.bar(
                    x1, db_diff,
                    color=COLORS[hi],
                    width=width,
                    # fill=False,
                    edgecolor='black'
            )
            plt.ylabel("dB")

    handles, labels = [], []
    plt.figure()
    for i, head in enumerate(postfix):
        if head == "":
            head = "None"
        else:
            head = head[1:]

        handles.append(Line2D([0, 0], [0, 0], color=COLORS[i], linewidth=5))
        labels.append(head)

    fig.legend(handles, labels, loc='lower right')
    fig.suptitle(
            f"Volume peaks of actions to ambient average volume. Higher=Better",
            fontsize=20,
    )
    plt.figure(fig)
    "Assign labels"
    for i in range(5):  # 5 subplots now
        plt.subplot(2, 3, i + 1)
        ax = plt.gca()

        if group_headphones:
            # ax.set_xticks([j + 1.5 for j in range(9)])
            ax.set_xticklabels([f"{h}" for h in headsets], rotation=30)
        else:
            ax.set_xticks([j + 0.5 for j in range(len(stack_intervals))])
            ax.set_xticklabels(
                    [
                            f"{inte}ms" if isinstance(inte, int) else "Raw Peak" for inte in
                            stack_intervals
                    ],
                    rotation=30)

    plt.tight_layout()
    fig.savefig(f"spread_stacked_{long}.png")


if __name__ == "__main__":
    # three_one_plot()
    # compare_spreads()
    # #plot_smooth_graph([1, 5, 15, 25])
    plot_spectograms([1, 5])
