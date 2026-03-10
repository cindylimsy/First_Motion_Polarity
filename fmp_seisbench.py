from typing import Any, Optional, Dict
from obspy import Stream, Trace, UTCDateTime 

import numpy as np
import torch
import torch.nn as nn

from seisbench.models.base import WaveformModel
import warnings
import pandas as pd
import matplotlib.pyplot as plt

class FM_model(WaveformModel):

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["stride"] = (_annotate_args["stride"][0], 1) # When applied to data > 4 secs in length

    def __init__(
        self,
        in_channels=1,
        classes=3,
        component="Z",
        phases="UDK",
        eps=1e-10,
        sampling_rate=100,
        pred_sample=200,
        original_compatible=True,
        filter_args=["bandpass"],
        filter_kwargs={'freqmin': 1, 'freqmax': 20, 'zerophase': False},
        **kwargs,
    ):
        
        super().__init__(
            # citation=citation,
            output_type="point",
            component_order=component,
            in_samples=400,
            pred_sample=pred_sample,
            sampling_rate=sampling_rate,
            labels=phases,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.eps = eps
        self.original_compatible = original_compatible
        self.filter_args = filter_args
        self.filter_kwargs = filter_kwargs
        self._phases = phases
        if phases is not None and len(phases) != classes:
            raise ValueError(
                f"Number of classes ({classes}) does not match number of labels ({len(phases)})."
            )

        self.conv1 = nn.Conv1d(in_channels, 32, 21, padding=10)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-3) # Confirmed this is what original model uses
        self.conv2 = nn.Conv1d(32, 64, 15, padding=7)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-3)
        self.conv3 = nn.Conv1d(64, 128, 11, padding=5)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-3)

        self.fc1 = nn.Linear(6400, 512)
        self.bn4 = nn.BatchNorm1d(512, eps=1e-3)
        self.fc2 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512, eps=1e-3)
        self.fc3 = nn.Linear(512, classes)

        self.activation = torch.relu

        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x, logits=False):
        # Max normalization - confirmed this is what paper says
        x = x / (
            torch.max(
                torch.max(torch.abs(x), dim=-1, keepdims=True)[0], dim=-2, keepdims=True
            )[0]
            + self.eps
        )
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))

        if self.original_compatible:
            # Permutation is required to be consistent with the following fully connected layer
            x = x.permute(0, 2, 1)
        x = torch.flatten(x, 1)

        x = self.activation(self.bn4(self.fc1(x)))
        x = self.activation(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        if logits:
            return x
        else:
            if self.classes == 1:
                return torch.sigmoid(x)
            else:
                return torch.softmax(x, -1)
            
            
    @property
    def phases(self):
        if self._phases is not None:
            return self._phases
        else:
            return list(range(self.classes))


    def annotate_stream(
        self,
        stream,
        picktimes,
        window_around_picktime=250):
        """
        Function to produce seismic traces and probability traces for all stations 
        for a +-250 sample window around the pick (500 sample)

        Input: 
            steam of continuous data (>= 500 samples)
        Output: 
            cut_stream = 5 seconds of continuous data,
            prob_traces = corresponding probability traces cut +-2.5 seconds around the P picktime
        """

        half_window = window_around_picktime
        cut_stream = Stream()

        # making sure stream only has vertical (Z) component
        stream = stream.select(component="Z")

        # return only for stations with picktimes
        selected_stream = Stream(tr for tr in stream if tr.stats.station in list(picktimes.station))

        # use the list of picktimes to cut a +- 250 sample sliding window from the continuous data
        for sta, tp in zip(picktimes["station"], picktimes["time"]):
            # cut continuous data for this station
            selected_stream_sta = selected_stream.select(station=sta)
            if len(selected_stream_sta) < 1:
                continue
            elif len(selected_stream_sta) > 1:
                for tr in selected_stream_sta:
                    start = tp - (half_window/tr.stats.sampling_rate)
                    end = tp + (half_window/tr.stats.sampling_rate)
                    tr_cut = tr.copy()
                    tr_cut.trim(starttime=start, endtime=end)
                    cut_stream += tr_cut
            else:
                tr = selected_stream_sta[0]
                start = tp - (half_window/tr.stats.sampling_rate)
                end = tp + (half_window/tr.stats.sampling_rate)
                tr_cut = tr.copy()
                tr_cut.trim(starttime=start, endtime=end)
                cut_stream += tr_cut

        ### Pre-process
        cut_stream.detrend('demean')
        cut_stream.detrend('linear')
        cut_stream.taper(0.05)
        prob_traces = self.annotate(cut_stream,stride=1)

        return cut_stream, prob_traces

    def classify_mean_sliding_window(
        self,
        stream,
        picktimes,
        time_win = 0.25,
        ignore_unknown=True,
        ignore_unknown_warning_thresh = 0.2,
        min_class_accept_ratio = 3,
        plot=False):
        """
        Function to populate the picktimes dataframe with the mean probabilities of all classes (U,D,K) 
        for the +-250 sample window, the final polarity class label and its corresponding probability.

        Input: 
            stream = steam of continuous data (>= 500 samples),
            picktimes = dataframe containing 'station' columns and 'time' columns (P picktimes in UTCDateTime format)
            time_win = time window around the pick, set at 0.25 seconds default (+-0.25 seconds = 0.5 seconds window)
            ignore_unknown = option to force a label instead of the "Unknown" class
            ignore_unknown_warning_thresh = threshold to ignore the unknown class, if not, a warning is flagged for future user decision
            min_class_accept_ratio = minimum accept ratio between the Up and Down class probabilities 
            plot = plot switch to show figures (stream and probability traces for each station)
        Output: 
            picktimes_labelled = dataframe of picktimes with populated with 
            mean polarity probability traces (mean_probabilities_of_classes),
            final polarity class label of the mean sliding window (mean_polarity_class), 
            and its corresponding probability value (mean_polarity_probability) for each station,
            and a prediction warning (prediction_warning) if the prediction class is not very certain 
            (when the maximum class probability is < min_class_accept_ratio * other class probability)
        """
        # supress pandas warnings
        warnings.simplefilter(action='ignore')

        # get cut stream and cut prob traces and full probability traces
        cut_stream, prob_traces = self.annotate_stream(stream,picktimes)

        # defining new columns and class labels
        picktimes['mean_probabilities_of_classes'] = pd.Series([None] * len(picktimes), dtype=object)
        picktimes['mean_polarity_class'] = [np.nan] * len(picktimes)
        picktimes['mean_polarity_probability'] = [np.nan] * len(picktimes)
        picktimes['prediction_warning'] = [False] * len(picktimes)
        classes = ['U','D','K']

        for sta in picktimes.station:
            # reset flag warning for plotting
            flag_warning = False
            # get the 3 probability traces for that station
            sta_traces = prob_traces.select(station=sta)
            probs = []
            if len(sta_traces) < 3:
                print("Station missing traces:", sta)
                continue

            # take the mean of the time window (default set to +- 0.25 seconds) around the picktime (set at the middle of the window)
            for tr in sta_traces:
                mid_window_point = len(tr)/2
                start_window = mid_window_point - (tr.stats.sampling_rate*time_win)
                end_window = mid_window_point + (tr.stats.sampling_rate*time_win)
                start_window = int(start_window)
                end_window = int(end_window)
                probs.append(np.mean(tr.data[start_window:end_window]))

            picktimes.at[picktimes.loc[picktimes['station'] == sta].index[0],'mean_probabilities_of_classes'] = probs

            # sorted classes from highest to lowest probability (descending)
            sorted_idx = np.argsort(probs)[::-1]
            pol = classes[sorted_idx[0]]
            pol_prob = probs[sorted_idx[0]]

            if ignore_unknown:
                if pol == 'K':
                    # take second highest probability, if pol_prob > ignore_unknown_thresh and more than triple the next class (if min_class_accept_ratio=3)
                    pol = classes[sorted_idx[1]]
                    pol_prob = probs[sorted_idx[1]]
                    if pol_prob < ignore_unknown_warning_thresh:
                        picktimes.at[picktimes.loc[picktimes['station'] == sta].index[0],'prediction_warning'] = True
                        flag_warning = True
            
            # get probabilities for Up/Down (no Unknown) and compute max/min
            prob_U = probs[classes.index('U')]
            prob_D = probs[classes.index('D')]
            max_prob = max(prob_U, prob_D)
            min_prob = min(prob_U, prob_D)

            # if the maximum probability is less than the min_class_accept_ratio * minimum probability, flag a prediction warning (i.e. the classification is not very certain)
            if max_prob < min_class_accept_ratio * min_prob:
                picktimes.at[picktimes.loc[picktimes['station'] == sta].index[0],'prediction_warning'] = True

            picktimes.at[picktimes.loc[picktimes['station'] == sta].index[0], 'mean_polarity_class'] =  pol
            picktimes.at[picktimes.loc[picktimes['station'] == sta].index[0], 'mean_polarity_probability'] =  pol_prob

            if plot:
                fig, axs = plt.subplots(2, 1, figsize=(10, 8))

                # First subplot: Unfiltered data
                axs[0].plot(cut_stream.select(station=sta)[0].data)
                axs[0].set_xlabel('Samples')
                axs[0].set_ylabel('Amplitude')
                axs[0].axvline((picktimes.loc[picktimes['station'] == sta].time.iloc[0]-cut_stream.select(station=sta)[0].stats.starttime)*cut_stream.select(station=sta)[0].stats.sampling_rate, color='red') # plot picktime
                label_y_position = max(cut_stream.select(station=sta)[0].data)-(0.2*max(cut_stream.select(station=sta)[0].data))
                if flag_warning:
                    axs[0].text(1,label_y_position,sta,fontweight='bold',color='red')
                else:
                    axs[0].text(1,label_y_position,sta,fontweight='bold')
                axs[0].set_xlim([0,len(cut_stream.select(station=sta)[0])])

                # 2nd subplot: Probability trace plots
                axs[1].plot(prob_traces.select(station=sta)[0], label='UP')
                axs[1].plot(prob_traces.select(station=sta)[1], label='DOWN')
                axs[1].plot(prob_traces.select(station=sta)[2], label='UNKNOWN')
                axs[1].legend(fontsize=14,loc='upper right')
                axs[1].set_xlabel('Prediction Sample Number')
                axs[1].set_ylabel('Probability')
                axs[1].axvline(25,color='red')
                axs[1].axvline(75,color='red')
                axs[1].axvspan(25, 75, alpha=0.1,color='red')
                label_y_position = 0.8
                if flag_warning:
                    axs[1].text(1,label_y_position,'Label: ' + pol + ' (WARNING)',fontweight='bold',color='red')
                else:
                    axs[1].text(1,label_y_position,'Label: ' + pol,fontweight='bold')
                axs[1].set_ylim([0,1])
                axs[1].set_xlim([0,len(prob_traces.select(station=sta)[0])])
                plt.tight_layout()
                plt.show()

        picktimes_labelled = picktimes

        return picktimes_labelled